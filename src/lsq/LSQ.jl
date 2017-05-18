
# Full-dimensional local search quantization
using HDF5, Clustering

include("../utils.jl")
include("../initializations.jl")
include("../codebook_update.jl")
include("../encodings/encode_icm.jl")

function train_lsq{T <: AbstractFloat}(
  X::Matrix{T},         # d-by-n matrix of data points to train on.
  m::Integer,           # number of codebooks
  h::Integer,           # number of entries per codebook
  R::Matrix{T},         # init rotation
  B::Matrix{Int16},     # init codes
  C::Vector{Matrix{T}}, # init codebooks
  niter::Integer,       # number of optimization iterations
  ilsiter::Integer,     # number of ILS iterations to use during encoding
  icmiter::Integer,     # number of iterations in local search
  randord::Bool,        # whether to use random order
  npert::Integer,       # The number of codes to perturb
  V::Bool=false)        # whether to print progress

  # if V
  println("**********************************************************************************************");
  println("Doing local search with $m codebooks, $npert perturbations, $icmiter icm iterations and random order = $randord");
  println("**********************************************************************************************");
  # end

  d, n = size( X );

  # Update RX
  RX = R' * X;

  # Initialize C
  C = update_codebooks( RX, B, h, V, "lsqr" )

  # Apply the rotation to the codebooks
  for i = 1:m
    C[i] = R * C[i]
  end
  @printf("%3d %e \n", -2, qerror( X, B, C ));

  # Initialize B
  for i = 1:ilsiter
    B = encoding_icm( X, B, C, icmiter, randord, npert, V );
    @everywhere gc()
  end
  @printf("%3d %e \n", -1, qerror( X, B, C ));

  obj = zeros( Float32, niter );

  for iter = 1:niter

    obj[iter] = qerror( X, B, C  );
    @printf("%3d %e \n", iter, obj[iter]);

    # Update the codebooks
    C = update_codebooks( X, B, h, V, "lsqr" )

    # Update the codes with local search
    for i = 1:ilsiter
      B = encoding_icm( X, B, C, icmiter, randord, npert, V );
      @everywhere gc()
    end

  end

  # Get the codebook for norms
  CB = reconstruct(B, C);

  dbnorms = zeros(Float32, 1, n);
  for i = 1:n
     for j = 1:d
        dbnorms[i] += CB[j,i].^2;
     end
  end

  # Quantize the norms with plain-old k-means
  dbnormsq = kmeans(dbnorms, h);
  cbnorms  = dbnormsq.centers;

  # Add the dbnorms to the codes
  B_norms    = reshape( dbnormsq.assignments, 1,n )

  return C, B, cbnorms, B_norms, obj

end
