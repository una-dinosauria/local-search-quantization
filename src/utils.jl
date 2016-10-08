
using DistributedArrays

# For eval_recall
using Distances

# Quantize the norms of an encoding
function quantize_norms(
  B::Matrix{Int16},           # Codes
  C::Vector{Matrix{Float32}}, # Codebooks
  cbnorms::Vector{Float32})   # codebook for norms

  CB   = reconstruct( B, C )
  d, n = size( CB )
  _, h = size( C[1] )

  dbnormsB = Vector{Int16}(n)
  dists2norm = Vector{Float32}(h)

  @inbounds for i = 1:n
    ithnorm = 0
    @simd for j = 1:d
      ithnorm += CB[j,i].^2
    end

    for j = 1:h
      dists2norm[j] = (ithnorm - cbnorms[j]).^2
    end
    _, dbnormsB[i] = findmin( dists2norm )
  end

  return dbnormsB
end

# Compute dot products between codebooks
function comp_prod_C{T <: AbstractFloat}(
  C::Vector{Matrix{T}}) # codebooks

  m   = length(C)
  dps = Matrix{Matrix{T}}( m, m ) # dot products
  for i = 1:m, j = (i+1):m
    dps[i,j] = 2 * C[i]' * C[j]
  end
  return dps
end

############################################
## Functions that support codebook update ##
############################################

# Creates a sparse matrix out of codes
function sparsify_codes{T <: Integer}(
  B::Matrix{T}, # m-by-n matrix. Codes to sparsify
  h::Integer)   # Number of entries per codebook

  m, n   = size( B )
  ncodes = length( B )

  # Storing the indices of the sparse matrix.
  I = zeros(Int32, ncodes) # row indices
  J = zeros(Int32, ncodes) # column indices

  for i = 1:m
    I[ (i-1)*n+1 : i*n ] = 1:n
    J[ (i-1)*n+1 : i*n ] = vec(B[i,:]) + (i-1)*h
  end

  C = sparse(I, J, ones(Float32, ncodes), n, h*m)

  return C
end

# Transform the output of LSQR/SPGL1 into a vector of matrices again
function K2vec(
  K::Union{SharedMatrix{Float32}, Matrix{Float32}}, # d-by-(h*m) matrix with the new codebooks
  m::Integer, # Number of codebooks
  h::Integer) # Number of elements in each codebook

  assert( size(K,2) == m*h )

  C = Vector{Matrix{Float32}}( m )

  subdims = splitarray( 1:(h*m), m )
  for i = 1:m
    C[i] = sdata( K[ :, subdims[i] ] )
  end

  return C
end

###########################################
## Functions to get unaries and binaries ##
###########################################

# Get unaries terms
@everywhere function get_unaries{T <: AbstractFloat}(
  X::Matrix{T},         # data to get unaries from
  C::Vector{Matrix{T}}, # codebooks
  V::Bool=false)        # whether to print progress

  d, n = size( X )
  m    = length(C)
  _, h = size( C[1] )

  unaries = Vector{Matrix{T}}(m)
  #if V print("Computing unaries... "); st=time(); end


  @inbounds for i = 1:m
    unaries[i] = -2 * C[i]' * X
    sci        = diag( C[i]' * C[i] )
    #unaries[i] = broadcast(+, sci, unaries[i])

    ui = unaries[i]
    for j = 1:n
      @simd for k = 1:h
        ui[k, j] += sci[k]
      end
    end
  end

  #if V @printf("done in %.3f seconds.\n", time()-st); end
  return unaries
end

# Computes all the binaries for a set of codebooks
function get_binaries{T <: AbstractFloat}(
  C::Vector{Matrix{T}}) # codebooks

  m = length( C )

  ncbi     = sum(1:(m-1)) # The number of pairwise interactions
  binaries = Vector{Matrix{T}}( ncbi ) # We'll store the binaries here
  cbi      = zeros(Int32, 2, ncbi) # Store the pairwise indices here

  idx=1
  for i = 1:m
    for j = (i+1):m
      binaries[idx] = 2 * C[i]' * C[j]
      cbi[:,idx]    = [i;j]
      idx = idx + 1
    end
  end

  return binaries, cbi
end

# Split an array in equal parts
# if the number of x is a multiple of nparts, then each parts has the same number of
# elements.
# if that's not the case, then the n rest elements, e.t. the rest of the division, will be
# added to the first n parts. So the first n parts contain each 1 item more than the the
# parts from n + 1 to nparts
function splitarray(
  x::Union{UnitRange, Vector}, # Vector to split in equal parts
  nparts::Integer)             # Number of parts to split the vector on

  n       = length(x)
  perpart = div( n, nparts )
  xtra    = mod( n, nparts )
  #out     = cell( nparts )
  out     = Array{Any}(nparts)

  #fills the parts, which have 1 item more than the other parts,
  #so these parts have size perpart + 1
  glidx = 1
  for i = 1:xtra
    out[i] = x[ glidx : (glidx+perpart) ]
    glidx  = glidx+perpart+1
  end

  #fills the parts with the normal size
  for i = (xtra+1):nparts
    out[i] = x[ glidx : (glidx+perpart-1) ]
    glidx  = glidx+perpart
  end

  return out
end

# Split a matrix horizontally in equal parts
# if the number of x is a multiple of nparts, then each parts has the same number of
# elements.
# if that's not the case, then the n rest elements, e.t. the rest of the division, will be
# added to the first n parts. So the first n parts contain each 1 item more than the the
# parts from n + 1 to nparts
function splitMatrixh{F <: AbstractFloat}(
  x::Matrix{F}, # Vector to split in equal parts
  nparts::Integer)             # Number of parts to split the vector on

  out = Vector{Matrix{F}}(nparts)
  _,n = size(x)
  outidx = splitarray(1:n,nparts)
  for i = 1:nparts
    out[i] = x[:,outidx[i]]
  end

  return out, outidx
end

###############################################
## Functions that evaluate global objectives ##
###############################################

function reconstruct{T <: AbstractFloat}(
  B::Matrix{Int16},
  C::Vector{Matrix{T}})

  m, n = size( B )
  d, _ = size( C[1] )
  CB   = zeros( T, d, n )

  @inbounds for i = 1:m
    mcb = C[i] # Pick the mth codebook
    for j = 1:n
      codej = B[i, j] # Choose the code
      @simd for k = 1:d     # And copy it all
        CB[k, j] += mcb[k, codej]
      end
    end
  end

  return CB

end

@everywhere function veccost{T <: AbstractFloat}(
  X::Matrix{T},
  #B::Union{Matrix{Int16},SharedMatrix{Int16}},
  B::Matrix{Int16},
  C::Vector{Matrix{T}})

  d, n = size( X )
  m,_ = size(B)

  cost = zeros(T, n)
  CBi = zeros(T,d)
  @inbounds for i = 1:n

    for k = 1:m
      Ci = C[k]
      code = B[k, i] # Choose the code
      @simd for j = 1:d     # And copy it all
        CBi[j] += Ci[j, code]
      end
    end

    @simd for j = 1:d
      cost[i] += (CBi[j] - X[j,i]).^2
      CBi[j] = 0
    end
  end

  return cost

end

# Evaluates the objective function for full-dimensional codebooks
function qerror{T <: AbstractFloat}(
  X::Matrix{T},
  B::Matrix{Int16},
  C::Vector{Matrix{T}})

  d, n = size( X )
  m,_ = size(B)

  accum::T = 0
  CBi = zeros(T,d)

  @inbounds for i = 1:n

    for k = 1:m
      Ci = C[k]
      code = B[k, i] # Choose the code
      @simd for j = 1:d     # And copy it all
        CBi[j] += Ci[j, code]
      end
    end

    @simd for j = 1:d
      accum += (CBi[j] - X[j,i]).^2
      CBi[j] = 0
    end
  end

  return accum ./ n
end

# Evaluates the objective functions for non full-dimensional C
function qerror_pq{T1 <: AbstractFloat, T2 <: Integer}(
  X::Matrix{T1},
  B::Matrix{T2},
  C::Vector{Matrix{T1}})

  d, n = size( X )
  return qerror_opq( X, B, C, eye(T1, d) )
end

function qerror_opq{T1 <: AbstractFloat, T2 <: Integer}(
  X::Matrix{T1},
  B::Matrix{T2},
  C::Vector{Matrix{T1}},
  R::Matrix{T1})

  CB = similar( X )
  subdims = splitarray( 1:size(X,1), length(C) )
  for i=1:length(C)
    CB[ subdims[i], : ] = C[i][:, vec(B[i,:]) ]
  end

  return mean(sum((R*CB - X).^2, 1 ))
end
