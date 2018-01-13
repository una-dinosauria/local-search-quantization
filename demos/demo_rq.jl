
using Rayuela

include("../src/read/read_datasets.jl")

function demo_rq(
  dataset_name="SIFT1M",
  nread::Integer=Int(1e4)) # Increase this to 1e5 to use the full dataset

  # === Hyperparams ===
  m       = 7 # In LSQ we use m-1 codebooks
  h       = 256
  verbose = true
  nquery  = Int(1e4)
  knn     = Int(1e3) # Compute recall up to
  b       = Int( log2(h) * m )
  niter   = 25

  # === OPQ initialization ===
  x_train              = read_dataset(dataset_name, nread )
  d, _                 = size( x_train )
  C, B, train_error = Rayuela.train_rq(x_train, m, h, niter, verbose)
  @printf("Error after RQ is %e\n", train_error)

  # === LSQ train ===
  ilsiter = 8
  icmiter = 4
  randord = true
  npert   = 4
  cpp     = true

  C, B, cbnorms, B_norms, obj = Rayuela.train_lsq(x_train, m, h, R, B, C, niter, ilsiter, icmiter, randord, npert, cpp, verbose)
  cbnorms = vec( cbnorms[:] )

  # === Encode the base set ===
  nread_base   = Int(1e6)
  x_base       = read_dataset(dataset_name * "_base", nread_base)
  B_base       = convert(Matrix{Int16}, rand(1:h, m, nread_base))

  ilsiter_base = 4 # LSQ-16 in the paper
  B = encoding_icm(x_base, B_base, C, ilsiter_base, icmiter, randord, npert, cpp, verbose)
  base_error = qerror(x_base, B_base, C)
  @printf("Error in base is %e\n", base_error)

  # Compute and quantize the database norms
  B_base_norms = quantize_norms( B_base, C, cbnorms )
  db_norms     = vec( cbnorms[ B_base_norms ] )

  # === Compute recall ===
  x_query = read_dataset(dataset_name * "_query", nquery, verbose)
  gt      = read_dataset(dataset_name * "_groundtruth", nquery, verbose)
  if dataset_name == "SIFT1M" || dataset_name == "GIST1M"
    gt = gt .+ 1
  end
  gt           = convert(Vector{UInt32}, gt[1,1:nquery])
  B_base       = convert(Matrix{UInt8}, B_base-1)
  B_base_norms = convert(Vector{UInt8}, B_base_norms-1)

  print("Querying m=$m ... ")
  @time dists, idx = linscan_lsq(B_base, x_query, C, db_norms, eye(Float32, d), knn)
  println("done")

  idx = convert(Matrix{UInt32}, idx);
  rec = eval_recall(gt, idx, knn)

end

# train
demo_rq()
