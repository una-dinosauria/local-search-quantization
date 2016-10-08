# Read a set of vectors stored in the bvec format (int + n * float)
# The function returns a set of output uint8 vector (one vector per column)
#
# Syntax:
#   v = bvecs_read (filename)        -> read all vectors
#   v = bvecs_read (filename, n)      -> read n vectors
#   v = bvecs_read (filename, [a b]) -> read the vectors from a to b (indices starts from 1)

function bvecs_read(
  bounds::UnitRange,
  filename::AbstractString="./data/sift1b/bigann_base.bvecs")

  @assert bounds.start >= 1

  # open the file and count the number of descriptors
  open(filename, "r") do fid

    # Read the vector size
    d = read(fid, Int32, 1)
    @assert length(d) == 1
    @show vecsizeof = 1 * 4 + d[1]

    # Get the number of vectrors
    seekend(fid)
    vecnum = position(fid) / vecsizeof

    # compute the number of vectors that are really read and go in starting positions
    n = bounds.stop - bounds.start + 1
    seekstart(fid)
    skip(fid, (bounds.start - 1) * vecsizeof)

    # read n vectors
    v = read(fid, UInt8, vecsizeof * n)
    v = reshape(v, vecsizeof, n)

    # Check if the first column (dimension of the vectors) is correct
    @assert sum( v[1,2:end] .== v[1, 1]) == n - 1
    @assert sum( v[2,2:end] .== v[2, 1]) == n - 1
    @assert sum( v[3,2:end] .== v[3, 1]) == n - 1
    @assert sum( v[4,2:end] .== v[4, 1]) == n - 1
    v = v[5:end, :]

    return v

  end
end

function bvecs_read(n::Integer, filename::AbstractString)
  return bvecs_read(1:n,filename)
end

function bvecs_read(n::Integer)
  return bvecs_read(1:n)
end
