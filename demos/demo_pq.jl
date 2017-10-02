
using Rayuela

include("../src/read/read_datasets.jl")
include("../src/linscan/Linscan.jl")

function demo_pq(
  dataset_name="SIFT1M",
  ntrain::Integer=Int(1e5))

  # === Hyperparams ===
  m       = 8
  h       = 256
  maxiter = 25
  verbose = true
  nquery  = Int(1e4)
  nbase   = Int(1e6)
  knn     = Int(1e3) # Compute recall up to
  b       = Int( log2(h) * m )

  # Load data
  x_train = read_dataset(dataset_name, ntrain)
  x_base  = read_dataset(dataset_name * "_base", nbase)
  x_query = read_dataset(dataset_name * "_query", nquery, verbose)
  gt      = read_dataset( dataset_name * "_groundtruth", nquery, verbose )
  if dataset_name == "SIFT1M" || dataset_name == "GIST1M"
    gt = gt .+ 1
  end
  gt = convert( Vector{UInt32}, gt[1,1:nquery] )

  # === Train ===
  C, B, train_error = train_pq(x_train, m, h, maxiter, verbose)
  @printf("Error in training is %e\n", train_error)

  # === Encode the base set ===
  B_base     = quantize_pq( x_base, C, verbose )
  base_error = qerror_pq( x_base, B_base, C )
  @printf("Error in base is %e\n", base_error)

  # === Compute recall ===
  println("Querying m=$m ... ")
  @time dists, idx = linscan_pq( convert( Matrix{UInt8}, B_base-1 ), x_query[:,1:nquery], C, b, knn )
  println("done")

  rec = eval_recall( gt, idx, knn )

end # function train_process

# train
demo_pq()
