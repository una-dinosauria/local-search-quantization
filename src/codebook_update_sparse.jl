
using MATLAB # To call spgl1

# include("utils.jl")

##################################################
## Codebook update using SPGL1 and thresholding ##
##################################################

function update_codebooks_spgl1(
  X::Matrix{Float32}, # d-by-n matrix to update codebooks on
  B::Matrix{Int16},   # m-by-n matrix. X encoded
  h::Integer,         # number of entries per codebook
  tau::Float64,       # \tau argument in SPGL1. Sets sparsity.
  prevC::Vector{Matrix{Float32}}, # The previous set of codebooks, to use as init
  spgl1_path::AbstractString, # Path to spgl1
  V::Bool=false)      # whether to print progress

  if V print("Doing SPGL1 codebook update... "); st=time(); end

  d, n   = size(X);
  m, _   = size(B);
  C = sparsify_codes( B, h );
  K = SharedArray(Float32, d, size(C,2));

  prevK =  convert( Matrix{Float64}, hcat( prevC... ) );

  # Convert C and X for matlab-friendlines
  C = convert(SparseMatrixCSC{Float64,Int64}, C);
  X = convert( Matrix{Float64}, X );

  # Copy values to matlab
  @mput X
  @mput C
  @mput tau
  @mput prevK
  @mput spgl1_path

  @matlab addpath( spgl1_path )

  mat"options = spgSetParms()"
  mat"options.verbosity = 1"

  @time begin
  mat"
    Xt = X';
    prevKt = prevK';

    [d, n] = size( X );

    [n, mh] = size( C );
    A = @(x, mode) sparse_lsq_fun( C, x, n, d, mode );

    try
      [K, ~, ~, ~] = spgl1(A, Xt(:), tau, [], prevKt(:), options);
    catch
      warning('spgl1 failed for some reason. Redoing without init')
      [K, ~, ~, ~] = spgl1(C, Xt(:), tau, [], [], options);
    end

    K = reshape( K, size(C,2), d );
    K = K';
    "
  end

  @mget K

  K = convert(Matrix{Float32}, K);
  new_C = K2vec( K, m, h );

  if V @printf("done in %.2f seconds\n", time()-st); end

  return new_C;

end # function update_codebooks

function update_codebooks_spgl1_threshold(
  X::Matrix{Float32}, # d-by-n matrix to update codebooks on
  B::Matrix{Int16},   # m-by-n matrix. X encoded
  h::Integer,         # number of entries per codebook
  tau::Float64,       # \tau argument in SPGL1. Sets sparsity
  prevC::Vector{Matrix{Float32}}, # The previous set of codebooks, to use as init
  S::Integer, # The number of highest non-zero entries to keep
  spgl1_path::AbstractString, # Path to spgl1
  V::Bool=false)      # whether to print progress

  d, n   = size(X);
  m, _   = size(B);

  # Call the regular SPGL1 function
  C = update_codebooks_spgl1( X, B, h, tau, prevC, spgl1_path, V );
  K = hcat( C... );

  @printf("%d non-zero elements after update\n", sum( K .!= 0 ))

  # Keep the S max elements
  highest = sortperm( abs(K[:]), rev=true );

  nk = length( K );
  for i = S+1:nk
    K[ highest[i] ] = 0;
  end

  C = K2vec( K, m, h );

end
