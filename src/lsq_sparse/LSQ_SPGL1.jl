
include("../utils.jl")
include("../initializations.jl")
include("../codebook_update_sparse.jl")
include("../encodings/encode_icm.jl")

function train_lsq_sparse{T <: AbstractFloat}(
  X::Matrix{T},      # d-by-n matrix of data points to train on.
  m::Integer,     # number of codebooks
  h::Integer,     # number of entries per codebook
  niter::Integer, # number of optimization iterations
  ilsiter::Integer,   # number of ILS iterations to use during encoding
  icmiter::Integer,   # number of iterations in local search
  randord::Bool,      # whether to use random order
  npert::Integer,     # The number of codes to perturb
  S::Integer,         # Number of non-zeros allowed in the codebooks (S)
  tau::Float64,       # The values of tau to use for all the dimensions
  B::Matrix{Int16},
  Cinit::Vector{Matrix{T}},
  R::Matrix{T},
  spgl1_path::AbstractString, # Path to spgl1
  V::Bool=true)      # whether to print progress

  # if V
  println("**********************************************************************************************");
  println("Doing local search with $m codebooks, $npert perturbations, $icmiter icm iterations and random order = $randord");
  println("**********************************************************************************************");
  # end

  d, n = size( X );

  # Create CB
  CB = similar( X );

  # Update RX
  RX = R' * X;

  # Make the codebooks to full-dimensional
  C  = Vector{Matrix{T}}(m)

  subdims = splitarray( 1:d, m );
  for i = 1:m
    C[i] = zeros( Float32, d, h );
    C[i][subdims[i],:] = Cinit[i];
  end

  obj = qerror( RX, B, C );
  @printf("Warm start error: %e\n", obj);

  # For making pretty plots on output
  l0s  = zeros(Float32, niter);
  l1s  = zeros(Float32, niter);
  objs = zeros(Float32, niter);

  # Initialize C
  C = update_codebooks_spgl1_threshold( RX, B, h, tau, C, S, spgl1_path, V );
  l1n = sum( abs( vcat(C...)) );
  spC = sparse( vcat(C...));
  @printf("%d non-zero elements. l1 norm is %e \n", nnz(spC), l1n)

  obj = qerror( RX, B, C );
  @printf("%3d %e \n", -1, obj);

  # Initialize B
  for i = 1:ilsiter
    @time B = encoding_icm( RX, B, C, icmiter, randord, npert, V );
  end
  obj = qerror( RX, B, C );

  @printf("%3d %e \n", -1, obj);

  obj     = Inf;
  objlast = Inf;

  for iter = 1:niter

    @show iter, niter

    objlast = obj;
    obj = qerror( RX, B, C  );
    @printf("%3d %e (%e better) \n", iter, obj, objlast - obj);

    # Update the codebooks
    C = update_codebooks_spgl1_threshold( RX, B, h, tau, C, S, spgl1_path, V );
    l1n = sum( abs(vcat(C...)) );
    spC = sparse( vcat(C...));
    @printf("%d non-zero elements. l1 norm is %e \n", nnz(spC), l1n)

    # Update the codes with local search
    for i = 1:ilsiter
      @time B = encoding_icm( RX, B, C, icmiter, randord, npert, V );
    end

    # Book-keeping
    l0s[iter]  = nnz(spC);
    l1s[iter]  = l1n;
    objs[iter] = obj;

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

  obj = qerror( RX, B, C );
  @printf("%3d %e \n", niter+1, obj);

  return C, B, R, obj, cbnorms, objs;

end
