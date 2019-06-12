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
    H = FlexibleHessian(S, nonzero_indices)
    y = polish_solution(x, H, [Y; -Y])
    # Correct the indices of H for the elements that the polishing flipped the sign of x
    for i in length(H.indices)
        idx = H.indices[i]
        if idx < n && y[idx] < 0
            H.indices[i] += n
        elseif idx > n && y[idx - n] > 0
            H.indices[i] -= n
        end
    end
    return y, H
end

function polish_solution(x::Vector{T}, H::FlexibleHessian{T, Tf}, W::Matrix{T}=zeros(T, 0)) where {T, Tf}
    if length(W) == 0
        v_max = eigs(H.H_small, which=:SR, nev=1, tol=1e-7)[2][:]
    else
        U = W[H.indices, :]
        F = qr(U, Val(true))
        L = LinearMap{Float64}(x -> project!(H.H_small*project!(x, U, F), U, F), size(H.H_small, 1); issymmetric=true)
        v_max = eigs(L, which=:SR, nev=1, tol=1e-7)[2][:]
    end
    x .= 0
    x[H.indices] .= v_max
    n = Int(length(x)/2)
    return x[1:n] - x[n+1:end]
end

function sparse_pca(S::Tf, gamma::T, y_init::Vector{T}, H=nothing; Y=zeros(0, 0), kwargs...) where {T, Tf}
    W = [Y; -Y]
    x_init = [max.(y_init, 0); -min.(y_init, 0)]
    data = solve_sparse_pca(S, gamma, x_init, H; W=W, kwargs...)
    if true # Polish?
        return polish_solution(copy(data.x), data.H, data.W), data
    else
        n = Int(length(x)/2)
        return data.x[1:n] - data.x[n+1:end], data
    end
end

function binary_search(S, nz, Y=zeros(0, 0); verbosity=1)
    # return sparsify(randn(size(S, 1)), nz), 1 # Dummy output for debbuging
    t_init = @elapsed x_init, H = QPnorm.get_initial_guess(S, Int(nz); Y=Y)
    println("Time required by the initialization step: ", t_init)
    println("Initial variance:", dot(x_init, S*sparse(x_init)))
    H = nothing
    high = norm(x_init, 1)
    low = norm(x_init, 1)/3

    n = length(x_init)
    x_warm = x_init
    max_iter = 30
    t = 0
    for i = 1:max_iter
        gamma = (high - low)/2 + low
        t += @elapsed y, data = QPnorm.sparse_pca(S, gamma, x_warm, H; Y=Y, verbosity=verbosity, printing_interval=5000, max_iter=10000);
        H = data.H
        x_warm = data.x[1:n] - data.x[n+1:end]
        nonzeros = sum(abs.(y) .> 1e-7)
        println("Nonzeros: ", nonzeros, " γ: [", high, ", ", low, "]")
        if nonzeros == nz || i == max_iter
            println("Found at iteration:", i)
            return y, data, t
        elseif nonzeros > nz
            high = gamma
        elseif nonzeros < nz
            low = gamma
        end
    end
end

function fake_binary_search(S, nz, Y=zeros(size(S, 1), 0); verbosity=1)
    return sparsify(randn(size(S, 1)), nz), 1 # Dummy output for debbuging
end