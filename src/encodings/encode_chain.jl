
@everywhere function encode_viterbi!{T <: AbstractFloat}(
  CODES::SharedMatrix{Int16},  # out. Where to save the result
  X::Matrix{T},                # in. d-by-n matrix to encode
  C::Vector{Matrix{T}},        # in. m-long vector with d-by-h codebooks
  binaries::Vector{Matrix{T}}, # in. Binary terms
  IDX::UnitRange{Int64})       # in. Index to save the result

  # Get unaries
  unaries = get_unaries( X, C );

  h, n = size( unaries[1] );
  m    = length( binaries ) + 1;

  # We need a matrix to keep track of the min and argmin
  mincost = zeros(T, h, m );
  minidx  = zeros(Integer, h, m );

  # Allocate memory for brute-forcing each pair
  cost = zeros( T, h );

  U = zeros(T, h, m)

  minv = typemax(T);
  mini = 1;

  uidx = 1;
  @inbounds for idx = IDX # Loop over the datapoints

    # Put all the unaries of this item together
    for i = 1:m
      ui = unaries[i]
      @simd for j = 1:h
        U[j,i] = ui[j,uidx];
      end
    end

    # Forward pass
    for i = 1:(m-1) # Loop over states

      # If this is not the first iteration, add the precomputed costs
      if i > 1
        @simd for j = 1:h
          U[j,i] += mincost[j,i-1];
        end
      end

      bb = binaries[i];
      for j = 1:h # Loop over the cost of going to j
        @simd for k = 1:h # Loop over the cost of coming from k
          ucost   =  U[k, i]; # Pay the unary of coming from k
          bcost   = bb[k, j]; # Pay the binary of going from j-k
          cost[k] = ucost + bcost;
        end

        # Writing my own findmin
        minv = cost[1]
        mini = 1
        for k = 2:h
          costi = cost[k]
          if costi < minv #|| minv!=minv
            minv = costi
            mini = k
          end
        end

        mincost[j, i] = minv;
         minidx[j, i] = mini;
      end
    end

    @simd for j = 1:h
      U[j,m] += mincost[j,m-1];
    end

    _, mini = findmin( U[:,m] );

    # Backward trace
    backpath = [ mini ];
    for i = (m-1):-1:1
      push!( backpath, minidx[ backpath[end], i ] );
    end

    # Save the inferred code
    CODES[:, idx] = reverse( backpath );

    uidx = uidx + 1;
  end # for idx = IDX
end


"Function to call that encodes a dataset using dynamic programming"
function encoding_viterbi(
  X::Matrix{Float32},         # d-by-n matrix. Data to encode
  C::Vector{Matrix{Float32}}, # m-long vector with d-by-h codebooks
  V::Bool=false)              # whether to print progress

  d, n = size( X );
  m    = length( C );

  # Compute binary tables
  binaries = Vector{Matrix{Float32}}(m-1);
  for i = 1:(m-1)
    binaries[i] = 2 * C[i]' * C[i+1];
  end
  CODES = SharedArray{Int16,2}((m, n));

  if nworkers() == 1
    encode_viterbi!( CODES, X, C, binaries, 1:n );
  else
    paridx = splitarray( 1:n, nworkers() );
    @sync begin
      for (i,wpid) in enumerate(workers())
        @async begin
          Xw = X[:,paridx[i]]
          remotecall_wait(encode_viterbi!, wpid, CODES, Xw, C, binaries, paridx[i]);
        end
      end
    end
  end

  return sdata(CODES);
end
