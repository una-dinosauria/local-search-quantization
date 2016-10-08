
include("../src/read/read_datasets.jl")
include("../src/utils.jl")
include("../src/pq/PQ.jl");
include("../src/linscan/Linscan.jl");

function demo_pq(
  dataset_name="SIFT1M",
  nread::Integer=Int(1e4)) # Increase this to 1e5 to use the full dataset

  # === Hyperparams ===
  m       = 8
  h       = 256
  verbose = true
  nquery  = Int(1e4)
  knn     = Int(1e3) # Compute recall up to
  b       = Int( log2(h) * m )


  # === Train ===
  x_train           = read_dataset(dataset_name, nread )
  C, B, train_error = train_pq(x_train, m, h, verbose)
  @printf("Error in training is %e\n", train_error)

  # === Encode the base set ===
  nread_base = Int(1e6)
  x_base     = read_dataset(dataset_name * "_base", nread_base )
  B_base     = quantize_pq( x_base, C, verbose )
  base_error = qerror_pq( x_base, B_base, C )
  @printf("Error in base is %e\n", base_error)

  # === Compute recall ===
  x_query = read_dataset( dataset_name * "_query", nquery, verbose )
  gt      = read_dataset( dataset_name * "_groundtruth", nquery, verbose )
  if dataset_name == "SIFT1M" || dataset_name == "GIST1M"
    gt = gt + 1;
  end
  gt = convert( Vector{UInt32}, gt[1,1:nquery] )
  B_base  = convert( Matrix{UInt8}, B_base-1 )

  print("Querying m=$m ... ");
  @time dists, idx = linscan_pq( B_base, x_query[:,1:nquery], C, b, knn )
  println("done");

  rec = eval_recall( gt, idx, knn );

end # function train_process

# train
demo_pq()
