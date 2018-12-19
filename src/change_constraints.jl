function remove_constraint!(data, idx::Int)
    constraint_idx = data.working_set[idx]
    ignored_constraints_add_row!(data, data.A[constraint_idx, :], data.b[constraint_idx])
    prepend!(data.ignored_set, constraint_idx)
    deleteat!(data.working_set, idx)
    remove_constraint!(data.F, idx)
end

function add_constraint!(data, idx::Int)
    constraint_idx = data.ignored_set[idx]
    add_constraint!(data.F, data.A[constraint_idx, :])
    ignored_constraints_remove_row!(data, idx)
    deleteat!(data.ignored_set, idx)
    append!(data.working_set, constraint_idx)
end

function ignored_constraints_add_row!(data, a::AbstractVector, b::AbstractFloat)
    l = length(data.ignored_set)
    data.A_shuffled[data.m - l, :] = a
    data.b_shuffled[data.m - l] = b

    data.A_ignored = view(data.A_shuffled, data.m - l:data.m, :)
    data.b_ignored = view(data.b_shuffled, data.m - l:data.m)
end

function ignored_constraints_remove_row!(data, idx::Int)
    l = length(data.ignored_set)
    start = data.m - l + 1
    @inbounds for j in 1:data.n, i in start+idx-1:-1:start+1
        data.A_shuffled[i, j] = data.A_shuffled[i - 1, j]
    end
    @inbounds for i in start+idx-1:-1:start+1
        data.b_shuffled[i] = data.b_shuffled[i - 1]
    end
    data.A_ignored = view(data.A_shuffled, start+1:data.m, :)
    data.b_ignored = view(data.b_shuffled, start+1:data.m)
end
