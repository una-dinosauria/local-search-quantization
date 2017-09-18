
import Rayuela

include("../src/read/read_datasets.jl")
include("../src/linscan/Linscan.jl")

# === Hyperparams ===
m       = 8 # Number of codebooks
h       = 256 # Number of entries per codebook
niter   = 25  # Number of iterations for training
verbose = true # Print progress for the user
ntrain, nbase, nquery  = Int(1e5), Int(1e6), Int(1e4) # Train, base and query size
knn     = Int(1e3) # Compute recall up to
b       = Int(log2(h) * m) # Number of bits used per vector
dataset_name="SIFT1M"

# === Load data ===
x_train = read_dataset(dataset_name, ntrain)
x_base  = read_dataset(dataset_name * "_base", nbase)
x_query = read_dataset(dataset_name * "_query", nquery, verbose)
gt      = read_dataset( dataset_name * "_groundtruth", nquery, verbose )
if dataset_name == "SIFT1M" || dataset_name == "GIST1M"
  gt = gt .+ 1
end
gt = convert( Vector{UInt32}, gt[1,1:nquery] )

# # === Product Quantizer ===
# # Train
# C, B, train_error = Rayuela.train_pq(x_train, m, h, niter, verbose)
# @printf("Error in training is %e\n", train_error)
# # Encode the base set
# B_base     = Rayuela.quantize_pq( x_base, C, verbose )
# base_error = Rayuela.qerror_pq( x_base, B_base, C )
# @printf("Error in base is %e\n", base_error)
# # Compute recall
# B_base  = convert( Matrix{UInt8}, B_base-1 )
# print("Querying m=$m ... ")
# @time dists, idx = linscan_pq( B_base, x_query[:,1:nquery], C, b, knn )
# println("done")
# rec = eval_recall( gt, idx, knn );


# === Optimized Product Quantizer ===
# Train
C, B, R, train_error = Rayuela.train_opq(x_train, m, h, niter, "natural", verbose)
@printf("Error in training is %e\n", train_error[end])
# Encode the base set
B_base     = Rayuela.quantize_opq( x_base, R, C, verbose )
base_error = Rayuela.qerror_opq( x_base, B_base, C, R )
@printf("Error in base is %e\n", base_error)
# Compute recall
B_base  = convert( Matrix{UInt8}, B_base-1 )
print("Querying m=$m ... ")
@time dists, idx = linscan_opq( B_base, x_query[:,1:nquery], C, b, R, knn )
println("done")
rec = eval_recall( gt, idx, knn )
