
using Rayuela

include("../src/read/read_datasets.jl")

function demo_rq(
  dataset_name="SIFT1M",
  nread::Integer=Int(1e5)) # Increase this to 1e5 to use the full dataset

  # === Hyperparams ===
  m       = 8 # In LSQ we use m-1 codebooks
  h       = 256
  verbose = true
  nquery  = Int(1e4)
  nbase   = Int(1e6)
  knn     = Int(1e3) # Compute recall up to
  b       = Int( log2(h) * m )
  niter   = 25

  # Load data
  x_train = read_dataset(dataset_name, nread)
  d, _    = size( x_train )
  x_base  = read_dataset(dataset_name * "_base", nbase)
  x_query = read_dataset(dataset_name * "_query", nquery, verbose)
  gt      = read_dataset( dataset_name * "_groundtruth", nquery, verbose )
  if dataset_name == "SIFT1M" || dataset_name == "GIST1M"
    gt = gt .+ 1
  end
  gt = convert( Vector{UInt32}, gt[1,1:nquery] )

  # === RQ train ===
  C, B, obj = Rayuela.train_ervq(x_train, m, h, niter, verbose)
	norms_B, norms_C = get_norms_codebook(B, C)

  # === Encode the base set ===
  B_base, _ = Rayuela.quantize_ervq(x_base, C, verbose)
  base_error = qerror(x_base, B_base, C)
  @printf("Error in base is %e\n", base_error)

  # Compute and quantize the database norms
  B_base_norms = quantize_norms( B_base, C, norms_C )
  db_norms     = vec( norms_C[ B_base_norms ] )

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
