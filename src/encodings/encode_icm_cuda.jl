using CUDAdrv, CuArrays

include("../utils.jl")
include("../read/read_datasets.jl")
include("../initializations.jl")
include("../encodings/cuda/cudaUtilsModule.jl")

function get_new_B(
  B::Matrix{Int16},
  m::Int,
  n::Int)

  newB = Matrix{Int16}(m, n); # codes
  @inbounds @simd for j = 1:m*n
    newB[j] = B[j];
  end

  return newB
end

"Encodes a database with ILS in cuda"
function encode_icm_cuda_single(
  RX::Matrix{Float32},         # in. The data to encode
  B::Matrix{Int16},            # in. Initial list of codes
  C::Vector{Matrix{Float32}},  # in. Codebooks
  ilsiters::Vector{Int64},     # in. ILS iterations to record Bs and obj function. Its max is the total number of iteration we will run.
  icmiter::Integer,            # in. Number of ICM iterations
  npert::Integer,              # in. Number of entries to perturb
  randord::Bool,               # in. Whether to randomize the order in which nodes are visited in ILS
  V::Bool=false)               # in. Whether to print progress

  d, n = size( RX )
  m    = length( C )
  _, h = size( C[1] )

  # Number of results to keep track of
  nr   = length( ilsiters )

  # Make space for the outputs
  Bs   = Vector{Matrix{Int16}}(nr)
  objs = Vector{Float32}(nr)

  # === Compute binary terms (products between all codebook pairs) ===
  binaries, cbi = get_binaries( C )
  _, ncbi       = size( cbi )

  # Create a transposed copy of the binaries for cache-friendliness
  binaries_t = similar( binaries )
  for j = 1:ncbi
    binaries_t[j] = binaries[j]'
  end

  # Create an index from codebook pairs to indices
  cbpair2binaryidx   = zeros(Int32, m, m)
  for j = 1:ncbi
    cbpair2binaryidx[ cbi[1,j], cbi[2,j] ] = j
  end

  dev = CuDevice(0)
  ctx = CuContext(dev)

  # Initialize the cuda module, and choose the GPU
  gpuid = 0
  CudaUtilsModule.init( gpuid, "src/encodings/cuda/cudautils.ptx" )
  # CUDArt.device( devlist[gpuid] )

  # === Create a state for random number generation ===
  if V; @printf("Creating %d random states... ", n); tic(); end
  d_state = CUDAdrv.Mem.alloc( n*64 )
  CudaUtilsModule.setup_kernel( cld(n, 1024), 1024, Cint(n), d_state )

  CUDAdrv.synchronize(ctx)
  if V; @printf("done in %.2f seconds\n", toq()); end

  # Measure time for encoding
  if V; tic(); end

  # Copy X and C to the GPU
  d_RX = CuArrays.CuArray(RX)
  d_C  = CuArrays.CuArray(cat(2, C... ))

  # === Get unaries in the gpu ===
  d_prevcost = CuArrays.CuArray{Cfloat}(n)
  d_newcost  = CuArrays.CuArray{Cfloat}(n)

  # CUBLAS.cublasCreate_v2( CUBLAS.cublashandle )
  d_unaries   = Vector{CuArrays.CuArray{Float32}}(m)
  d_codebooks = Vector{CuArrays.CuArray{Float32}}(m)
  for j = 1:m
    d_codebooks[j] = CuArrays.CuArray(C[j])
    # -2 * C' * X
    d_unaries[j] = CuArrays.BLAS.gemm('T', 'N', -2.0f0, d_codebooks[j], d_RX)
    # d_unaries[j] = -2.0f0 * d_codebooks[j]' * d_RX <-- thus runs out of memory real fast

    # Add self-codebook interactions || C_{i,i} ||^2
    CudaUtilsModule.vec_add( n, (1,h), d_unaries[j].buf, CuArrays.CuArray(diag( C[j]' * C[j] )).buf, Cint(n), Cint(h) )
  end

  # === Get binaries to the GPU ===
  d_binaries  = Vector{CuArrays.CuArray{Float32}}( ncbi )
  d_binariest = Vector{CuArrays.CuArray{Float32}}( ncbi )
  for j = 1:ncbi
    d_binaries[j]  = CuArrays.CuArray( binaries[j] )
    d_binariest[j] = CuArrays.CuArray( binaries_t[j] )
  end

  # Allocate space for temporary results
  bbs = Vector{Matrix{Cfloat}}(m-1)

  # Initialize the previous cost
  prevcost = veccost( RX, B, C )
  IDX = 1:n;

  # For codebook i, we have to condition on these codebooks
  to_look      = 1:m
  to_condition = zeros(Int32, m-1, m)
  for j = 1:m
    tmp = collect(1:m)
    splice!( tmp, j )
    to_condition[:,j] = tmp
  end

  # Loop for the number of requested ILS iterations
  for i = 1:maximum( ilsiters )

    # @show i, ilsiters
    # @time begin

    to_look_r      = to_look
    to_condition_r = to_condition

    # Compute the cost of the previous assignments
    # CudaUtilsModule.veccost(n, (1, d), d_RX, d_C, CuArray( convert(Matrix{Cuchar}, (B')-1) ), d_prevcost, Cint(m), Cint(n))
    CudaUtilsModule.veccost2(n, (1, d), d_RX.buf, d_C.buf, CuArrays.CuArray( convert(Matrix{Cuchar}, (B')-1) ).buf, d_prevcost.buf, Cint(d), Cint(m), Cint(n))
    CUDAdrv.synchronize(ctx)
    prevcost = Array( d_prevcost )

    newB = copy(B)

    # Randomize the visit order in ICM
    if randord
      to_look_r      = randperm( m )
      to_condition_r = to_condition[:, to_look_r]
    end

    d_newB = CuArrays.CuArray( convert(Matrix{Cuchar}, newB-1 ) )

    # Perturn npert entries in each code
    CudaUtilsModule.perturb( n, (1,m), d_state, d_newB.buf, Cint(n), Cint(m), Cint(npert) )

    newB = Array( d_newB )
    Bt   = newB'
    d_Bs = CuArrays.CuArray( Bt )

    CUDAdrv.synchronize(ctx)

    # Run the number of ICM iterations requested
    for j = 1:icmiter

      kidx = 1;
      # Loop through each MRF node
      for k = to_look_r

        lidx = 1;
        for l = to_condition_r[:, kidx]
          # Determine the pairwise tables that we'll use in this conditioning
          if k < l
            binariidx = cbpair2binaryidx[ k, l ]
            bbs[lidx] = binaries[ binariidx ]
          else
            binariidx = cbpair2binaryidx[ l, k ]
            bbs[lidx] = binaries_t[ binariidx ]
          end
          lidx = lidx+1;
        end

        # Transfer pairwise tables to the GPU
        d_bbs = CuArrays.CuArray( convert(Matrix{Cfloat}, cat(2,bbs...)) )
        # Sum binaries (condition) and minimize
        CudaUtilsModule.condition_icm3(
          n, (1, h),
          d_unaries[k].buf, d_bbs.buf, d_Bs.buf, Cint(k-1), Cint(m), Cint(n) )
        CUDAdrv.synchronize(ctx)

        kidx = kidx + 1;
      end # for k = to_look_r
    end # for j = 1:icmiter

    newB = Array( d_Bs )
    newB = convert(Matrix{Int16}, newB')
    newB .+= 1

    # Keep only the codes that improved
    CudaUtilsModule.veccost2(n, (1, d), d_RX.buf, d_C.buf, d_Bs.buf, d_newcost.buf, Cint(d), Cint(m), Cint(n))
    CUDAdrv.synchronize(ctx)

    newcost = Array( d_newcost )

    areequal = newcost .== prevcost
    if V; @printf(" ILS iteration %d/%d done. ", i, maximum(ilsiters)); end
    if V; @printf("%5.2f%% new codes are equal. ", 100*sum(areequal)/n ); end

    arebetter = newcost .< prevcost
    if V; @printf("%5.2f%% new codes are better.\n", 100*sum(arebetter)/n ); end

    newB[:, .~arebetter] = B[:, .~arebetter]
    B = copy( newB )
    # end # @time

    # Check if this # of iterations was requested
    if i in ilsiters

      ithidx = find( i .== ilsiters )[1]

      # Compute and save the objective
      obj = qerror( RX, B, C )
      # @show obj
      objs[ ithidx ] = obj
      # @show size(B)
      Bs[ ithidx ] = B

    end # end if i in ilsiters
  end # end for i=1:max(ilsiters)

  # CUBLAS.cublasDestroy_v2( CUBLAS.cublashandle )
  CudaUtilsModule.finit()

  destroy!(ctx)
  # end # do devlist

  if V; @printf(" Encoding done in %.2f seconds\n", toq()); end

  return Bs, objs
end

# OLD API
# function encode_icm_cuda(
#   RX::Matrix{Float32},
#   B::Matrix{Int16},
#   C::Vector{Matrix{Float32}},
#   cbnorms::Vector{Float32},
#   ilsiters::Vector{Int64},
#   icmiter::Integer,
#   npert::Integer,
#   randord::Bool,
#   qdbnorms::Bool
#   )
#
#   return Bs, objs
# end

"Encodes a database with ILS in cuda"
function encode_icm_cuda(
  RX::Matrix{Float32},         # in. The data to encode
  B::Matrix{Int16},            # in. Initial list of codes
  C::Vector{Matrix{Float32}},  # in. Codebooks
  ilsiters::Vector{Int64},     # in. ILS iterations to record Bs and obj function. Its max is the total number of iteration we will run.
  icmiter::Integer,            # in. Number of ICM iterations
  npert::Integer,              # in. Number of entries to perturb
  randord::Bool,               # in. Whether to randomize the order in which nodes are visited in ILS
  nsplits::Integer=2,          # in. Number of splits of the data (for limited memory GPUs)
  V::Bool=false)

  # TODO check that splits >= 1
  if nsplits == 1
    Bs, objs =  encode_icm_cuda_single(RX, B, C, ilsiters, icmiter, npert, randord, V)
    return Bs, objs
  end

  # Split the data
  d, n = size( RX )
  splits = splitarray(1:n, nsplits)
  nr = length(ilsiters)

  Bs   = Vector{Matrix{Int16}}(nr)
  objs = zeros(Float32,nr)

  # Create storage space for the codes
  for i = 1:nr; Bs[i] = Matrix{Int16}(size(B)); end

  # Run encoding in the GPU for each split
  for i = 1:nsplits
    aaBs, _ = encode_icm_cuda_single(RX[:,splits[i]], B[:,splits[i]], C, ilsiters, icmiter, npert, randord, V)
    for j = 1:nr
      # Save the codes
      Bs[j][:,splits[i]] = aaBs[j]
    end
  end

  # Compute the cost again
  for j = 1:nr
    objs[j] = qerror(RX, Bs[j], C)
  end

  return Bs, objs
end
