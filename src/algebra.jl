using Arpack
import Base.getindex, Base.size

function circle_line_intersections(a1::T, a2::T, b::T, r::T) where T
    """
    Calculates the points intersecting points between the 2d plane defined
    as a1*x + a2*y = b
    and the circle x^2 + y^2 = r^2
    Returns x1, y1, x2, y2 such that [x1; y1] and [x2; y2] are the two intersecting points.
    If no intersection(s) exist then NaN values are returned.
    Reference: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """
    # Normalize, i.e. make [a1; a2] unit norm
    norm_a = sqrt(a1^2 + a2^2)
    if norm_a < 1e-11  # Very badly scaled problem
        @warn "Badly scaled circe-line"
        return T(NaN), T(NaN), T(NaN), T(NaN)
    end
    a1 /= norm_a; a2 /= norm_a; b /= norm_a

    # First calculate a point (x, y) on the line
    # and dx, dy such that (x + dx, y + dy) lies also on the line.
    if abs(a1) > abs(a2) # Avoid dividing with something small
        x = (b - a2)/a1
        y = one(T)

        dx = -a2/a1
        dy = one(T)
    else
        x = one(T)
        y = (b - a1)/a2

        dx = one(T)
        dy = -a1/a2
    end

    # Make (dx, dy) unit norm
    norm_d = sqrt(dx^2 + dy^2)
    dx /= norm_d; dy /= norm_d

    D = x*(y + dy) - (x + dx)*y
    #= Allow for slightly infeasible solutions
    if abs(r)/abs(D) <= one(T)
        @show abs(r)/abs(D) - 1
        if abs(r)/abs(D) >= one(T) - 1e-11
            D = r
        end
    end
    =#
    if r >= abs(D)
        c = sqrt(r^2 - D^2)

        x1 = D*dy; y1 = -D*dx
        x2 = x1;  y2 = y1;

        δ1 = dx*c
        if dy < 0; δ1 = -δ1 end
        δ2 = abs(dy)*c

        x1 += δ1; y1 += δ2
        x2 -= δ1; y2 -= δ2

        return x1, y1, x2, y2
    else
        return T(NaN), T(NaN), T(NaN), T(NaN)
    end
end

function trs_robust(P::AbstractArray{T}, q::AbstractVector{T}, r::T; tol=0.0, v0=zeros((0,)), kwargs...) where T
    n = length(q)
    if norm(q) <= 1e-10
        if n > 20
            l, v = eigs(P, which=:SR, nev=1, tol=1e-11)
        else
            l, v = eigen(P)
        end
        v = v[:, 1]/norm(v[:, 1])*r
        return [v -v], TRS.TRSinfo(true, 0, 0, [l[1]; l[1]])
    end

    if n < 15
        return trs_boundary_small(P, q, r; kwargs...)
    else
        try
            return trs_boundary(P, q, r; tol=tol, v0=v0, kwargs...)
        catch e
            if isa(e, LAPACKException)
                try
                    @warn "Error #1:", typeof(e), " - trying again"
                    return trs_boundary(P, q, r; kwargs...)
                catch
                    #    @show e
                    @warn "Error #2:", e, " - switching to direct"
                    return trs_boundary_small(P, q, r; kwargs...)
                end
            else
                @warn "Error:", e, " - switching to direct"
                return trs_boundary_small(P, q, r; kwargs...)
            end
        end
    end
end

mutable struct FlexibleHessian{T, Tf}
    S::Tf
    H::Symmetric{T, SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int},UnitRange{Int}}, false}}
    data::Matrix{T}  # That's where P is viewing into
    indices::Vector{Int}

    function FlexibleHessian(S::Tf, indices::Vector{Int}) where {Tf}
        m = length(indices)
        T = Float64 #typeof(getindex(S, 1, 1))
        data = zeros(T, 2*m, 2*m)
        unwrapped_indices = mod.(indices .- 1, size(S, 1)) .+ 1
        H = view(data, 1:m, 1:m)
        H .= -getindex(S, unwrapped_indices, unwrapped_indices)
        unwrap!(H, size(S, 1), indices, indices)
        new{T, Tf}(S, Symmetric(H), data, copy(indices))
    end
end

function add_column!(FP::FlexibleHessian{T, Tf}, idx::Int) where {T, Tf}
    m = length(FP.indices)

    # Expand FP.data if it is already full
    if size(FP.data, 1) == m
        new_data = zeros(T, 2*m, 2*m)
        new_data[1:m, 1:m] = FP.data
        FP.data = new_data
    end

    append!(FP.indices, idx)
    n = size(FP.S, 1)
    unwrapped_indices = mod.(FP.indices .- 1, n) .+ 1
    FP.data[1:m+1, m+1:m+1] = -getindex(FP.S, unwrapped_indices, [mod(idx - 1, n) + 1])
    unwrap!(view(FP.data, 1:m+1, m+1), n, FP.indices, [idx])

    FP.H = Symmetric(view(FP.data, 1:m+1, 1:m+1))
    # W = [FP.P_full -FP.P_full; -FP.P_full FP.P_full]
    # show(stdout, "text/plain", W[FP.indices, FP.indices]); println()
    # show(stdout, "text/plain", FP.P); println()
end

function remove_column!(FP::FlexibleHessian{T, Tf}, position::Int) where {T, Tf}
    m = length(FP.indices)
    @inbounds for j = position:m-1, i = 1:position-1
        FP.data[i, j] = FP.data[i, j+1]
    end
    @inbounds for j = position:m-1, i = position:j
        FP.data[i, j] = FP.data[i+1, j+1]
    end
    deleteat!(FP.indices, position)
    FP.H = Symmetric(view(FP.data, 1:m-1, 1:m-1))
end

struct CovarianceMatrix{T}
    D::T # An kxn matrix-like object containing k n-dimensional observations with zero mean
         # The matrix-like type {T} of D must allow for indexing (of the form D[:, i])
         # and (normal/transposed) multiplication (with *)
    μ::Vector{Float64} # equal to mean(D, dims=1)
    L_deflate::Matrix{Float64}
    R_deflate::Matrix{Float64}

    function CovarianceMatrix(D::T, L_deflate=zeros(0), R_deflate=zeros(0)) where {T}
        # @assert all(abs.(mean(D, dims=1)) .<= 1e-9) "Please make sure the dataset has zero mean observations"
        if length(L_deflate) == 0 || length(R_deflate) == 0
            L_deflate = zeros(Float64, size(D, 2), 1)
            R_deflate = zeros(Float64, size(D, 2), 1)
        end
        new{T}(D, reshape(mean(D, dims=1), size(D, 2)), L_deflate, R_deflate)
    end
end

function Base.:(*)(S::CovarianceMatrix{T}, x::AbstractVector{Tf}) where {T, Tf}
    y = Vector(S.D*x .- dot(S.μ, x))
    y = S.D'*y - sum(y)*S.μ
    for k = 1:size(S.L_deflate, 2)
        y .-= (S.L_deflate[:, k:k]*(S.R_deflate[:, k:k]'*x) + S.R_deflate[:, k:k]*(S.L_deflate[:, k:k]'*x))/2
    end
    return y
end

function size(S::CovarianceMatrix{T}, idx::Int) where {T, Tf}
    return size(S.D, 2)
end

function unwrap!(X, n::Int, i_indices::Vector{Int}, j_indices::Vector{Int})
    idx = 1;
    @inbounds for j in j_indices, i in i_indices
        if (i <= n && j > n) || (i > n && j <= n)
            X[idx] = -X[idx]
        end
        idx += 1
    end
    return X
end

function getindex(S::CovarianceMatrix, i_indices::Vector{Int}, j_indices::Vector{Int})
    Dj = S.D[:, j_indices]
    sum_dj = sum(Dj, dims=1)'
    μ_j = S.μ[j_indices, :]
    if i_indices === j_indices
        Di = Dj
        sum_di = sum_dj
        μ_i = μ_j
    else
        Di = S.D[:, i_indices]
        sum_di = sum(Di, dims=1)'
        μ_i = S.μ[i_indices, :]
    end
    y = Dj'*Di - sum_dj*μ_i' - μ_j*sum_di' + size(S.D, 1)*μ_j*μ_i'

    # Deflation
    idx = 1;
    @inbounds for j in j_indices, i in i_indices
        @inbounds for k = 1:size(S.L_deflate, 2)
            y[idx] -= (S.L_deflate[i, k]*S.R_deflate[j, k] + S.L_deflate[j, k]*S.R_deflate[i, k])/2
        end
        idx += 1
    end

    return y
end

function sparse_mul(S::CovarianceMatrix{T}, x::Vector{Tf}) where {T, Tf}
    n = Int(length(x)/2)
    w = x[1:n] - x[n+1:end]
    y = Vector(_sparse_mul(S.D, x) .- dot(S.μ, w))
    y = S.D'*y - sum(y)*S.μ
    for k = 1:size(S.L_deflate, 2)
        y .-= (S.L_deflate[:, k:k]*(S.R_deflate[:, k:k]'*w) + S.R_deflate[:, k:k]*(S.L_deflate[:, k:k]'*w))/2
    end
    return [y; -y]
end

function sparse_mul(S::AbstractMatrix{Tm}, x::Vector{Tv}) where {Tm, Tv}
    y = _sparse_mul(S, x)
    return [y; -y]
end

function _sparse_mul(S::AbstractMatrix{Tm}, x::Vector{Tv}) where {Tm, Tv}
    n = size(S, 2)
    sparse_diff = sparse(x[1:n] - x[n+1:end])
    return S*sparse_diff
end