@everywhere using Distributions

# Encode using iterated conditional modes
@everywhere function encode_icm_fully!{T <: AbstractFloat}(
  B::Union{Matrix{Int16},SharedMatrix{Int16}},  # in/out. Initialization, and the place where the results are saved.
  X::Matrix{T},                 # in. d-by-n data to encode
  C::Vector{Matrix{T}},         # in. m-long vector with d-by-h codebooks
  binaries::Vector{Matrix{T}},  # in. The binary terms
  cbi::Matrix{Int32},           # in. 2-by-ncbi. indices for inter-codebook interactions
  niter::Integer,               # in. number of iterations for block-icm
  randord::Bool,                # in. whether to randomize search order
  npert::Integer,               # in. number of codes to perturb per iteration
  IDX::UnitRange{Int64},        # in. Index to save the result
  V::Bool)                      # in. whether to print progress

  # Compute unaries
  unaries = get_unaries( X, C, V );

  h, n = size( unaries[1] );
  m, _ = size( B );

  ncbi = length( binaries );

  # Create a transposed copy of the binaries
  binaries_t = similar( binaries );
  for i = 1:ncbi
    binaries_t[i] = binaries[i]';
  end

  # Create an index from codebook pairs to indices
  cbpair2binaryidx   = zeros(Int32, m, m);
  for i = 1:ncbi
    cbpair2binaryidx[ cbi[1,i], cbi[2,i] ] = i;
  end

  # For codebook i, we have to condition on these codebooks
  to_look      = 1:m;
  to_condition = zeros(Int32, m-1, m);
  for i = 1:m
    tmp = collect(1:m);
    splice!( tmp, i );
    to_condition[:,i] = tmp;
  end

  # Make the order random
  if randord
    to_look      = randperm( m );
    to_condition = to_condition[:, to_look];
  end

  # Preallocate some space
  bb = Matrix{T}( h, h );
  ub = Matrix{T}( h, n );

  # Sample the indices to perturb in each code
  pertidx  = Matrix{Integer}( npert, n );
  for i = 1:n
    sampleidx = sample(1:m, npert, replace=false, ordered=true);
    for j = 1:npert
      pertidx[j, i] = sampleidx[j];
    end
  end
  pertvals = rand( 1:h, npert, n );

  # Perturb the solutions
  for i = 1:n
    for j = 1:npert
      B[ pertidx[j,i], IDX[i] ] = pertvals[j,i];
    end
  end

  @inbounds for i=1:niter # Do the number of passed iterations

    # This is the codebook that we are updating (i.e., the node)
    jidx = 1;
    for j = to_look
      # Get the unaries that we will work on
      uj = unaries[j];
      @simd for k = 1:h*n
        ub[k] = uj[k];
      end

      # These are all the nodes that we are conditioning on (i.e., the edges to 'absorb')
      for k = to_condition[:, jidx];

        # Determine the pairwise interactions that we'll use (for cache-friendliness)
        if j < k
          binariidx = cbpair2binaryidx[ j, k ];
          bb = binaries[ binariidx ];
        else
          binariidx = cbpair2binaryidx[ k, j ];
          bb = binaries_t[ binariidx ];
        end

        # Traverse the unaries, absorbing the appropriate binaries
        for l=1:n
          codek = B[k, IDX[l]];
          @simd for ll = 1:h
            ub[ll, l] += bb[ ll, codek ];
          end
        end
      end #for k=to_condition

      # Once we are done conditioning, traverse the absorbed unaries to find mins
      inidx = 1;
      for idx=IDX
        minv = ub[1, inidx]
        mini = 1
        for k = 2:h
          ubi = ub[k, inidx]
          if ubi < minv
            minv = ubi
            mini = k
          end
        end

        B[j, idx] = mini;
        inidx = inidx + 1;
      end

      jidx = jidx + 1;


    end # for j=to_look
  end # for i=1:niter

end


# Encode a full dataset
function encoding_icm{T <: AbstractFloat}(
  X::Matrix{T},         # d-by-n matrix. Data to encode
  oldB::Matrix{Int16},  # m-by-n matrix. Previous encoding
  C::Vector{Matrix{T}}, # m-long vector with d-by-h codebooks
  niter::Integer,       # number of ICM iterations
  randord::Bool,        # whether to use random order
  npert::Integer,       # the number of codes to perturb
  V::Bool=false)        # whether to print progress

  d, n =    size( X )
  m    =  length( C )
  _, h = size( C[1] )

  # Compute binaries between all codebook pairs
  binaries, cbi = get_binaries( C )
  _, ncbi       = size( cbi )

  # Compute the cost of the previous assignments
  prevcost = veccost( X, oldB, C )

  if nworkers() == 1
    B = zeros(Int16, m, n)
  else
    B = SharedArray(Int16, m, n)
  end

  @inbounds @simd for i = 1:m*n
    B[i] = oldB[i]
  end

  if nworkers() == 1
    encode_icm_fully!( B, X, C, binaries, cbi, niter, randord, npert, 1:n, V );
    #encode_icm_cpp!( B, X, C, binaries, cbi, niter, randord, npert, 1:n, V );
  else
    paridx = splitarray( 1:n, nworkers() );
    @sync begin
      for (i,wpid) in enumerate(workers())
        @async begin
          Xw = X[:,paridx[i]];
          remotecall_wait(encode_icm_fully!, wpid, B, Xw, C, binaries, cbi, niter, randord, npert, paridx[i], V );
        end
      end
    end
  end
  B = sdata(B)

  # Keep only the codes that improved
  newcost = veccost( X, B, C )

  areequal = newcost .== prevcost;
  if V println("$(sum(areequal)) new codes are equal"); end

  arebetter = newcost .< prevcost;
  if V println("$(sum(arebetter)) new codes are better"); end

  B[:, .~arebetter] = oldB[:, .~arebetter]

  return B
end
