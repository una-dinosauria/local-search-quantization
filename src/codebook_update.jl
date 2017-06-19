
@everywhere using IterativeSolvers.lsqr

# Pulls K2vec and sparsify_codes
include("utils.jl")

# Update a dimension of a codebook using LSQR or LSMR
@everywhere function updatecb!(
  K::SharedMatrix{Float32},
  C::SparseMatrixCSC{Int32,Int32},
  X::Matrix{Float32},
  IDX::UnitRange{Int64},
  codebook_upd_method::AbstractString="lsqr")   # choose the codebook update method out of lsqr or lsmr

  if codebook_upd_method == "lsqr"
    for i = IDX
      K[i,:] = IterativeSolvers.lsqr( C, vec(X[i, :]) );
    end
  elseif codebook_upd_method == "lsmr"
    for i = IDX
      K[i,:] = IterativeSolvers.lsmr( C, vec(X[i, :]) );
    end
  else
    error("Codebook update method unknown: ", codebook_upd_method)
  end
end

@everywhere function updatecb!(
  K::SharedMatrix{Float32},
  C::SparseMatrixCSC{Float32,Int32},
  X::Matrix{Float32},
  IDX::UnitRange{Int64},
  codebook_upd_method::AbstractString="lsqr")   # choose the codebook update method out of lsqr or lsmr

  if codebook_upd_method == "lsqr"
    for i = IDX
      K[i,:] = IterativeSolvers.lsqr( C, vec(X[i, :]) );
    end
  elseif codebook_upd_method == "lsmr"
    for i = IDX
      K[i,:] = IterativeSolvers.lsmr( C, vec(X[i, :]) );
    end
  else
    error("Codebook update method unknown: ", codebook_upd_method)
  end
end

###############################################
## Codebook update for a fully-connected MRF ##
###############################################

function update_codebooks(
  X::Matrix{Float32}, # d-by-n matrix to update codebooks on.
  B::Matrix{Int16},   # m-by-n matrix. X encoded.
  h::Integer,         # number of entries per codebook.
  V::Bool=false,      # whether to print progress
  codebook_upd_method::AbstractString="lsqr")   # choose the codebook update method out of lsqr or lsmr

  if !(codebook_upd_method in ["lsmr", "lsqr"]); error("Codebook update method unknown"); end

  if V print("Doing " * uppercase(codebook_upd_method) * " codebook update... "); st=time(); end

  d, n   = size(X)
  m, _   = size(B)
  C = sparsify_codes( B, h )

  K = SharedArray(Float32, d, size(C,2))
  if nworkers() == 1
    updatecb!( K, C, X, 1:d )
  else
    paridx = splitarray( 1:d, nworkers() )
    @sync begin
      for (i, wpid) in enumerate( workers() )
        @async begin
          remotecall_wait( updatecb!, wpid, K, C, X, paridx[i], codebook_upd_method )
        end #@async
      end #for
    end #@sync
  end

  new_C = K2vec( K, m, h )

  if V @printf("done in %.3f seconds.\n", time()-st); end

  return new_C
end # function update_codebooks

function get_cbdims_chain(
  d::Integer, # The number of dimensions
  m::Integer) # The number of codebooks

  subdims = splitarray(1:d, m-1);
  odims = Vector{UnitRange{Integer}}(m);

  odims[1] =  subdims[1];
  for i = 2:m-1;
    odims[i] = subdims[i-1][1]:subdims[i][end];
  end
  odims[end] = subdims[end];

  return odims
end

# Update a dimension of a codebook using LSQR
@everywhere function updatecb_struct!(
  K::SharedMatrix{Float32},
  C::SparseMatrixCSC{Float32,Int32},
  X::Matrix{Float32},
  dim2C, #::Vector{Bool},
  subcbs,
  IDX::UnitRange{Int64} )

  for i = IDX
    rcbs      = cat( 1, subcbs[find(dim2C[i,:])]... );
    K[i,rcbs] = lsqr( C[:,rcbs], vec(X[i, :]) );
  end
end

function update_codebooks_generic(
  X::Matrix{Float32},  # d-by-n. The data that was encoded.
  B::Union{Matrix{Int16},SharedArray{Int16,2}}, # d-by-m. X encoded.
  h::Integer,          # number of entries per codebooks
  odimsfunc::Function, # Function that says which dimensions each codebook has
  V::Bool=false)       # whether to print progress

  if V print("Doing LSQR codebook update... "); st=time(); end

  d, n   = size( X );
  m, _   = size( B );
  C = sparsify_codes( B, h );

  odims = odimsfunc(d, m);

  # Make a map of dimensions to codebooks
  dim2C = zeros(Bool, d, m);
  for i = 1:m; dim2C[ odims[i], i ] = true; end

  K = SharedArray(Float32, d, h*m);
  subcbs = splitarray(1:(h*m), m);

  if nworkers() == 1
    updatecb_struct!( K, C, X, dim2C, subcbs, 1:d );
  else
    paridx = splitarray( 1:d, nworkers() );
    @sync begin
      for (i, wpid) in enumerate( workers() )
        @async begin
          remotecall_wait(updatecb_struct!, wpid, K, C, X, dim2C, subcbs, paridx[i] );
        end #@async
      end #for
    end #@sync
  end

  new_C = K2vec( K, m, h );
  if V @printf("done in %.3f seconds.\n", time()-st); end

  return new_C
end

# Update codebooks in a chain
function update_codebooks_chain(
  X::Matrix{Float32}, # d-by-n. The data that was encoded.
  B::Union{Matrix{Int16},SharedArray{Int16,2}}, # d-by-m. X encoded.
  h::Integer,         # number of entries per codebook.
  V::Bool=false)      # whether to print progress

  return update_codebooks_generic(X, B, h, get_cbdims_chain, V);

end
