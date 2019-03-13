using LinearAlgebra
using Arpack
using LinearMaps

function sparsify_nonnegative(x::Vector{Float64}, nnz::Int)
    # Discard (i.e. make zero) all but the largest nnz (in absolute value) elements in x
    y = x/norm(x)
    sorted_indices = sortperm(y)

    indices_positive = sorted_indices[end-nnz+1:end]
    y_positive = max.(y[indices_positive], 0)
    indices_negative = sorted_indices[1:nnz]
    y_negative = max.(-y[indices_negative], 0)

    y .= 0
    if norm(y_positive) > norm(y_negative)
        y[indices_positive] .= y_positive
    else
        y[indices_negative] .= y_negative
    end
    y ./= norm(y)
    return y
end

function polish_nonzero!(data)
    polished_data = eTRS.solve_sparse_pca(-Matrix(data.H.H_small), # Covariance matrix
        Float64(length(data.x_nonzero)), # 1-norm constraint: we set it large enough so as it's inactive
        copy(data.x_nonzero); # Initial point
        printing_interval=5000,
        max_iter=10000)

    x = zeros(size(data.x))
    x[data.nonzero_indices] .= polished_data.x
    return x
end

function get_initial_guess_nonnegative(S, nnz; Y=zeros(0, 0))
    n = size(S, 1)
    if length(Y) == 0
        L = LinearMap(x -> S*x, n; issymmetric=true)
    else
        # Find rightmost eigenvector that is perpendicular to Y
        F = qr(Y, Val(true))
        L = LinearMap{Float64}(x -> project!(S*project!(x, Y, F), Y, F), n; issymmetric=true)
    end
    v_max = eigs(L, which=:LR, nev=1, tol=1e-7)[2][:]
    # Keep on the largest elements (we make the others zero)
    return sparsify_nonnegative(v_max, nnz)
end

function binary_search_nonnegative(S, nz, Y=zeros(0, 0); verbosity=2)
    @show @elapsed x_init = eTRS.get_initial_guess_nonnegative(S, nz; Y=Y)
    println("Initial variance:", dot(x_init, S*sparse(x_init)))
    H = nothing
    high = norm(x_init, 1)
    low = norm(x_init, 1)/3

    n = length(x_init)
    x_warm = x_init
    max_iter = 30
    for i = 1:max_iter
        gamma = (high - low)/2 + low
        data = eTRS.solve_sparse_pca(S, gamma, x_warm, H; W=Y, verbosity=verbosity, printing_interval=5000, max_iter=10000);
        x_warm = copy(data.x)
        H = data.H
        nonzeros = sum(abs.(data.x) .> 1e-7)
        println("Nonzeros: ", nonzeros, " γ: [", high, ", ", low, "]")
        if nonzeros == nz || i == max_iter
            println("Found at iteration:", i)
            y = polish_nonzero!(data)
            return y, data
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