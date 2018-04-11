include("../src/read/read_datasets.jl")
include("../src/utils.jl")
include("../src/pq/PQ.jl");
include("../src/lsq_sparse/LSQ_SPGL1.jl");
include("../src/linscan/Linscan.jl");

function demo_lsq_sparse(
  dataset_name="SIFT1M",
  nread::Integer=Int(1e4))

  # === Hyperparams ===
  m       = 7 # In LSQ we use m-1 codebooks
  h       = 256
  verbose = true
  nquery  = Int(1e4)
  knn     = Int(1e3) # Compute recall up to
  b       = Int( log2(h) * m )
  niter   = 10

  # === PQ initialization ===
  x_train           = read_dataset(dataset_name, nread )
  d, _              = size( x_train )
  C, B, train_error = train_pq(x_train, m, h, verbose)
  @printf("Error after PQ is %e\n", train_error)

  # === LSQ sparse train ===
  ilsiter = 8
  icmiter = 4
  randord = true
  npert   = 4
  randord = true
  S       = d*h # SLSQ1 in the paper. Use S = d*h + (d.^2) for SLSQ2
  tau     = 0.7 # 0.7 for SLSQ1. Use 0.9 for SLSQ2

  # Multiply tau times the l1 norm of the PQ solution
  taus    = zeros( d )
  subdims = splitarray( 1:d, m )
  for i = 1:m
    taus[ subdims[i] ] += sum( abs(C[i]), 2 ) .* tau
  end
  tau = sum( taus )

  spgl1_path = joinpath( pwd(), "matlab/spgl1")
  C, B, R, train_error, cbnorms, objs =
    train_lsq_sparse(x_train, m, h, niter, ilsiter, icmiter, randord, npert, S, tau,
    B, C, eye(Float32, d), spgl1_path)
  cbnorms = vec( cbnorms[:] )

  # === Encode the base set ===
  nread_base   = Int(1e6)
  x_base       = read_dataset(dataset_name * "_base", nread_base )
  B_base       = randinit(nread_base, m, h) # initialize B at random

  ilsiter_base = 16 # LSQ-16 in the paper
  for i = 1:ilsiter_base
    @printf("Iteration %02d / %02d\n", i, ilsiter_base)
    @time B_base = encoding_icm( x_base, B_base, C, icmiter, randord, npert, verbose )
  end
  base_error = qerror( x_base, B_base, C )
  @printf("Error in base is %e\n", base_error)

  # Compute and quantize the database norms
  B_base_norms = quantize_norms( B_base, C, cbnorms )
  db_norms     = vec( cbnorms[ B_base_norms ] )

  # === Compute recall ===
  x_query = read_dataset( dataset_name * "_query", nquery, verbose )
  gt      = read_dataset( dataset_name * "_groundtruth", nquery, verbose )
  if dataset_name == "SIFT1M" || dataset_name == "GIST1M"
    gt = gt + 1;
  end
  gt           = convert( Vector{UInt32}, gt[1,1:nquery] )
  B_base       = convert( Matrix{UInt8}, B_base-1 )
  B_base_norms = convert( Vector{UInt8}, B_base_norms-1 )

  print("Querying m=$m ... ")
  @time dists, idx = linscan_lsq( B_base, x_query, C, db_norms, eye(Float32, d), knn )
  println("done")

  idx = convert( Matrix{UInt32}, idx );
  rec = eval_recall( gt, idx, knn )

end

# train
demo_lsq_sparse()
