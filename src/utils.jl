using LinearAlgebra
using Arpack
using LinearMaps

function sparsify(x::Vector{T}, nnz::Int) where {T}
    # Discard (i.e. make zero) all but the largest nnz (in absolute value) elements in x
    y = x/norm(x)
    sorted_indices = sortperm(abs.(y))
    y[sorted_indices[1:end-nnz]] .= 0
    y ./= norm(y)
    return y
end

function get_initial_guess(S, nnz::Int; Y::Matrix{T}=zeros(0, 0)) where {T}
    n = size(S, 1)
    if isa(S, Matrix{T})
        S = Symmetric(S)
    end

    if length(Y) == 0
        L = LinearMap{Float64}(x -> S*x, n; issymmetric=true)
    else
        # Find rightmost eigenvector that is perpendicular to Y
        F = qr(Y, Val(true))
        L = LinearMap{Float64}(x -> project!(S*project!(x, Y, F), Y, F), n; issymmetric=true)
    end
    v_max = eigs(L, which=:LR, nev=1, tol=1e-7)[2][:]
    # Keep on the largest elements (we make the others zero)
    y_init = sparsify(v_max, nnz)
    # if length(Y) > 0; project!(y_init, Y, F); end
    # Polish solution
    x = [max.(y_init, 0, ); -min.(y_init, 0)]
    nonzero_indices = findall(x .>= 1e-9)
    H_nonzero = FlexibleHessian(S, nonzero_indices)
    return polish_solution(x, H_nonzero.H, [Y; -Y])
end

function polish_solution(x::Vector{T}, H::AbstractMatrix{T}, W::Matrix{T}=zeros(T, 0)) where {T}
    nonzero_indices = findall(x .>= 1e-9)
    @assert size(H, 1) == size(H, 2) == length(nonzero_indices)
    if length(W) == 0
        v_max = eigs(H, which=:SR, nev=1, tol=1e-7)[2][:]
    else
        U = W[nonzero_indices, :]
        F = qr(U, Val(true))
        L = LinearMap{Float64}(x -> project!(H*project!(x, U, F), U, F), size(H, 1); issymmetric=true)
        v_max = eigs(L, which=:SR, nev=1, tol=1e-7)[2][:]
    end
    x .= 0
    x[nonzero_indices] .= v_max
    n = Int(length(x)/2)
    return x[1:n] - x[n+1:end]
end

function sparse_pca(S, gamma::T, y_init::Vector{T}; kwargs...) where T
    data = solve_sparse_pca(S, gamma, y_init; kwargs...)
    if true # Polish?
        return polish_solution(copy(data.x), data.H_nonzero.H, data.W), data
    else
        n = Int(length(x)/2)
        return data.x[1:n] - data.x[n+1:end], data
    end
end