
#
#  Updates assignments, costs, and counts based on
#  an updated (squared) distance matrix
#
function update_assignments!{T<:AbstractFloat}(
    dmat::Matrix{T},            # in:  distance matrix (k x n)
    is_init::Bool,              # in:  whether it is the initial run
    assignments::Union{Vector{Int}, SubArray},   # out: assignment vector (n)
    costs::Vector{T},           # out: costs of the resultant assignment (n)
    counts::Vector{Int},        # out: number of samples assigned to each cluster (k)
    to_update::Vector{Bool},    # out: whether a center needs update (k)
    unused::Vector{Int})        # out: the list of centers get no samples assigned to it

    k::Int, n::Int = size(dmat)

    # re-initialize the counting vector
    fill!(counts, 0)

    if is_init
        fill!(to_update, true)
    else
        fill!(to_update, false)
        if !isempty(unused)
            empty!(unused)
        end
    end

    # process each sample
    @inbounds for j = 1 : n

        # find the closest cluster to the i-th sample
        a::Int = 1
        c::T = dmat[1, j]
        for i = 2 : k
            ci = dmat[i, j]
            if ci < c
                a = i
                c = ci
            end
        end

        # set/update the assignment
        if is_init
            assignments[j] = a
        else  # update
            pa = assignments[j]
            if pa != a
                # if assignment changes,
                # both old and new centers need to be updated
                assignments[j] = a
                to_update[a] = true
                to_update[pa] = true
            end
        end

        # set costs and counts accordingly
        costs[j] = c
        counts[a] += 1
    end

    # look for centers that have no associated samples

    for i = 1 : k
        if counts[i] == 0
            push!(unused, i)
            to_update[i] = false # this is handled using different mechanism
        end
    end
end

#
#  Update centers based on updated assignments
#
#  (specific to the case where samples are not weighted)
#
function update_centers!{T<:AbstractFloat}(
  x::Matrix{T},                   # in: sample matrix (d x n)
  assignments::Vector{Int},       # in: assignments (n)
  cweights::Vector{T},            # out: updated cluster weights (k)
  k::Integer)

  d::Int = size(x, 1)
  n::Int = size(x, 2)

  centers = zeros(T, d, k)

  # initialize center weights
  for i = 1:k
    cweights[i] = 0
  end

  for i = 1:n
    ci = assignments[i]
    cweights[ci] += 1
  end

  # @show sum( cweights .== 0 )

  # accumulate columns
  @inbounds for j = 1:n
    cj = assignments[j]
    1 <= cj <= k || error("assignment out of boundary.")

    for i = 1:d
      centers[i, cj] += x[i, j]
    end
  end

  # sum ==> mean
  for j = 1:k
    if cweights[j] != 0
      @inbounds cj::T = 1 / cweights[j]
      vj = view(centers,:,j)
      for i = 1:d
        @inbounds vj[i] *= cj
      end
    end
  end

  return centers

end
