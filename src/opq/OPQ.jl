
# Optimized Product Quantization. Adapted from Mohammad Norouzi's code.

using Distances
using Distributions # for sampling

include("kmeans.jl")
include("../pq/PQ.jl")
include("../utils.jl")

function quantize_opq{T <: AbstractFloat}(
  X::Matrix{T}, # d-by-n matrix of data points to quantize
  R::Matrix{T}, # d-by-d matrix. Learned rotation for X
  C::Vector{Matrix{T}},  # m-long array. Each entry is a d-by-h codebooks
  V::Bool=false)

  # Apply rotation and quantize as in PQ
  return quantize_pq( R'*X, C, V )
end

function train_opq{T <: AbstractFloat}(
  X::Matrix{T},         # d-by-n matrix of data points to train on.
  m::Integer,           # number of codebooks
  h::Integer,           # number of entries per codebook
  niter::Integer,       # number of optimization iterations
  init::AbstractString, # how to initialize the optimization
  V::Bool=false )    # wheter to print progress

  d, n = size( X )

  C = Vector{Matrix{T}}(m) # codebooks

  obj = zeros( Float32, niter+1 )

  # Number of bits in the final codes.
  nbits = log2(h) * m
  CB    = zeros(T, size(X))

  if init == "natural" # Initialize R with identity
    R = eye(T, d)
  elseif init == "random"
    R, _, _ = svd( randn( T, d, d ))
  else
    error("Intialization $init unknown")
  end

  RX = R' * X # Rotate the data

  subdims = splitarray( 1:d, m )

  # Initialization sampling RX
  for i = 1:m
    perm = sample(1:n, h, replace=false)
    C[i] = RX[ subdims[i], perm ]
  end

  costs     = zeros(Float32, n)
  counts    = zeros(Int, h)
  cweights  = zeros(Float32, h)
  to_update = zeros(Bool, h)
  unused    = Int[]

  # Initialize the codes -- B
  B = zeros( Int64, n, m )
  for i=1:m
    dmat = pairwise( SqEuclidean(), C[i], RX[subdims[i],:] )
    dmat = convert(Array{Float32}, dmat)
    update_assignments!( dmat, true, view( B,:,i ), costs, counts, to_update, unused  )

    CB[subdims[i],:] = C[i][:, B[:,i]]
  end

  for iter=0:niter

    obj[iter+1] = sum( (R*CB - X).^2 ) ./ n
    @printf("%3d %e   \n", iter, obj[iter+1])

    # update R
    U, S, V = svd(X * CB', thin=true)
    R = U * V'

    # update R*X
    RX = R' * X

    for i=1:m
      # update C
      C[i] = update_centers!( RX[subdims[i],:], B[:,i], cweights, h )

      # update B
      dmat = pairwise( SqEuclidean(), C[i], RX[subdims[i],:] )
      dmat = convert(Array{Float32}, dmat)
      update_assignments!( dmat, false, view(B,:,i), costs, counts, to_update, unused  )

      # update D*B
      CB[ subdims[i], : ] = C[i][:, B[:, i]]
    end # for i=1:m
  end # for iter=0:niter

  C = convert( Vector{Matrix{T}}, C )
  return C, B', R, obj
end
