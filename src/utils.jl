using LinearAlgebra
using Arpack
using LinearMaps

function sparsify(x::Vector{T}, nnz::Int) where {T}
    # Discard (i.e. make zero) all but the largest nnz (in absolute value) elements in x
    y = x/norm(x)
    sorted_indices = sortperm(abs.(y))
    y[sorted_indices[1:end-nnz]] .= 0
    # for i = nnz:length(sorted_indices)
        #y[sorted_indices[i]] = 0
    # end
    y ./= norm(y)
    return y
end

function project!(x, A::Matrix{T}, F::Factorization{T}) where {T}
    # Project x to the nullspace of A. F is the QR factorization of A.
    l = F\x
    x .-= A*l
end

function get_initial_guess(S, nnz::Int; Y::Matrix{T}=zeros(0, 0)) where {T}
    n = size(S, 1)
    if isa(S, Matrix{T})
        S = Symmetric(S)
    end

    if length(Y) == 0
        v_max = eigs(S, which=:LR, nev=1)[2][:]
    else
        # Find rightmost eigenvector that is perpendicular to Y
        F = qr(Y, Val(true))
        L = LinearMap{Float64}(x -> project!(S*project!(x, Y, F), Y, F), n; issymmetric=true)
        v_max = eigs(L, which=:LR, nev=1)[2][:]
    end
    # Keep on the largest elements (we make the others zero)
    y_init = sparsify(v_max, nnz)
    # if length(Y) > 0; project!(y_init, Y, F); end
    # Polish solution
    polish!(y_init, S; Y=Y)

    return y_init, norm(y_init, 1)
end

function polish!(y::Vector{T}, S; Y::Matrix{T}=zeros(0, 0)) where {T}
    nonzero_indices = findall(abs.(y) .> 1e-8)
    y[abs.(y) .<= 1e-8] .= 0

    # Define a smaller eigenvalue problem at the nonzero indices
    nzs = length(nonzero_indices)
    # Construct Snz := S[nonzero_indices, nonzero_indices]
    Snz = zeros(T, nzs, nzs)
    for j = 1:nzs, i = 1:j
        Snz[i, j] = getindex(S, nonzero_indices[i], nonzero_indices[j])
    end
    Snz = Symmetric(Snz)

    if length(Y) == 0
        v_max_nz = eigs(Snz, which=:LR, nev=1)[2][:]
    else
        Ynz = Y[nonzero_indices, :]
        Fnz = qr(Ynz, Val(true))
        L = LinearMap{T}(x -> project!(Snz*project!(x, Ynz, Fnz), Ynz, Fnz), nzs; issymmetric=true)
        v_max_nz = eigs(L, which=:LR, nev=1)[2][:]
    end
    y[nonzero_indices] = v_max_nz

    return y
end
