
using HDF5

include("../utils.jl")
include("../initializations.jl")
include("../codebook_update.jl")
include("../encodings/encode_chain.jl")

# Train a chain quantizer with viterbi decoding
function train_chainq{T <: AbstractFloat}(
  X::Matrix{T},             # d-by-n matrix of data points to train on.
  m::Integer,               # number of codebooks
  h::Integer,               # number of entries per codebook
  R::Matrix{T},             # Init rotation matrix
  B::Matrix{Int16},         # Init codes
  C::Vector{Matrix{T}},     # Init codebooks
  niter::Integer,           # number of optimization iterations
  V::Bool=false)            # whether to print progress

  d, n = size( X )
  obj  = zeros(Float32, niter+1)

  CB = zeros( Float32, size(X) )
  RX = R' * X

  # Initialize C
  C = update_codebooks_chain( RX, B, h, V )
  @printf("%3d %e\n", -2, qerror( RX, B, C ))

  # Initialize B
  B   = encoding_viterbi( RX, C, V )
  @printf("%3d %e\n", -1, qerror( RX, B, C ))

  for iter = 0:niter
    obj[iter+1] = qerror( RX, B, C  )
    @printf("%3d %e\n", iter, obj[iter+1])

    # update CB
    CB[:] = 0;
    for i = 1:m; CB += C[i][:, vec(B[i,:]) ]; end

    # update R
    U, S, VV = svd(X * CB', thin=true)
    R = U * VV';

    # update R*X
    RX = R' * X

    # Update the codebooks #
    C = update_codebooks_chain( RX, B, h, V )

    # Update the codes with lattice search
    B = encoding_viterbi( RX, C, V );

  end

  return C, B, R, obj
end
