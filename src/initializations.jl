
function randinit(
  n::Integer, # Number of elements to initialize
  m::Integer, # Number of codebooks
  h::Integer) # Number of elements per codebook
  B = convert( Matrix{Int16}, rand(1:h, m, n) );
  return B
end
