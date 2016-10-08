#module Product_quantization
using Clustering

#for computing distances
using Distances

#for computing the error with k-means
include("../utils.jl")
include("../opq/kmeans.jl")

# Quantize using PQ codebooks
function quantize_pq{T <: AbstractFloat}(
  X::Matrix{T},         # d-by-n. Data to encode
  C::Vector{Matrix{T}}, # codebooks
  V::Bool=false)        # whether to print progress

  d, n = size( X )
  m    = length( C )
  h    = size( C[1], 2 )

  B       = zeros(Int, m, n) # codes
  subdims = splitarray( 1:d, m )

  # auxiliary variables for update_assignments! function
  costs     = zeros(Float32, n)
  counts    = zeros(Int, h)
  to_update = zeros(Bool, h)
  unused    = Int[]

  for i = 1:m
    if V print("Encoding on codebook $i / $m... ") end

    # Find distances from X to codebook
    dmat = pairwise( SqEuclidean(), C[i], X[subdims[i],:] )
    dmat = convert(Array{T}, dmat)
    update_assignments!( dmat, true, view(B,i,:), costs, counts, to_update, unused )

    if V println("done"); end
  end
  return convert(Matrix{Int16}, B) # return the codes
end

# Train product quantization
function train_pq{T <: AbstractFloat}(
  X::Matrix{T},  # d-by-n. Data to learn codebooks from
  m::Integer,    # number of codebooks
  h::Integer,    # number of entries per codebook
  V::Bool=false) # whether to print progress

  d, n = size( X )

  C = Vector{Matrix{T}}(m); # codebooks

  B        = zeros(Int16, m, n); # codes
  subdims  = splitarray(1:d, m); # subspaces

  for i = 1:m
    if V print("Working on codebook $i / $m... "); end
    cluster = kmeans( X[ subdims[i],: ], h, init=:kmpp);
    C[i], B[i,:] = cluster.centers, cluster.assignments;

    if V
      subdim_cost = cluster.totalcost ./ n;
      nits        = cluster.iterations;
      converged   = cluster.converged;

      println("done.");
      println("  Ran for $nits iterations");
      println("  Error in subspace is $subdim_cost");
      println("  Converged: $converged")
    end
  end

  error = qerror_pq( X, B, C )
  return C, B, error
end

#end # module PQ
