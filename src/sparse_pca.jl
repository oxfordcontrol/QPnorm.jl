using LinearAlgebra
using Arpack
using LinearMaps
using SparseArrays
using Printf

mutable struct Data{T, Tf}
    """
    Data structure for the solution of the sparse pca problem, formulated as:
        min         ½x'Hx
        subject to  ‖x‖ = 1
                    x ≥ 0
                    sum(x) ≤ γ,
    with an active set algorithm.
    """
    x::Vector{T}  # Variable of the minimization problem
    x_nonzero::Vector{T}  # Nonzero variable
    x0::Vector{T} # x_nonzero - project(x_nonzero) where projection is onto the nullspace of the working constraints

    S::Tf # Covariance matrix of the dataset
    W::Matrix{T} # Linear orthogonality constraints
    gamma::T

    H_nonzero::FlexibleHessian{T, Tf} # Flexible hessian
    L::Matrix{T} # Matrix containing a subset of 
    F::LinearAlgebra.QRPivoted{T,Matrix{T}} # QR factorization of the linear equality constraints
    R::Matrix{T}

    nonzero_indices::Vector{Int}
    zero_indices::Vector{Int}
    gamma_active::Bool

    Xmin::Matrix{T} # Last TRS minimizer(s)

    # Mutlipliers
    λ::Vector{T} # Lagrange Multipliers for the -x <= 0 constraints
    μ::T # Lagrange Multiplier for the ||x|| = 1 constraint
    ν::T # Lagrange Multiplier for the sum(x) <= gamma constraint
    κ::Vector{T} # Lagrange mutliplier for the orthogonality constraint

    # Options
    tolerance::T
    verbosity::Int
    printing_interval::Int

    # Logging
    done::Bool
    iteration::Int
    residual::T
    trs_choice::Char
    grad_steps::Int
    eigen_steps::Int
    timings::DataFrame

    function Data(S::Tf, gamma::T, y_init::Vector{T}; kwargs...) where {T, Tf}
        x = [max.(y_init, 0); -min.(y_init, 0)]
        nonzero_indices = findall(x .>= 1e-9)
        H = FlexibleHessian(S, nonzero_indices)
        return Data(S, H, gamma, y_init; kwargs...)
    end

    function Data(S::Tf, H::FlexibleHessian{T, Tf}, gamma::T, y_init::Vector{T}; Y::Matrix{T}=zeros(T, 0, 0),
        verbosity=1, printing_interval=50, tolerance=1e-11, kwargs...) where {T, Tf}

        @assert !isa(S, Symmetric) "Do not pass the covariance matrix in Symmetric type."
        nonzero_indices = copy(H.indices)
        zero_indices = setdiff(1:2*length(y_init), nonzero_indices)
        if length(Y) == 0
            Y = zeros(T, size(S, 1), 1)
        else
            @assert norm(Y'*Y - I) < 1e-9 "Matrix of previous sparse principal vectors must be orthonormal."
        end

        @assert norm(Y'*y_init) < 1e-9 "Starting vector not perpendicular to previous principal vectors"
        if norm(y_init, 1) > gamma || norm(y_init) - 1 <= -1e-9
            # Scale y_init so that either the one-norm or two norm constraint is active
            y_init /= max(norm(y_init, 1)/gamma, norm(y_init, 2)) 
        end
        @assert norm(y_init) - 1 <= 1e-9
        x = [max.(y_init, 0, ); -min.(y_init, 0)]
        gamma_active = (sum(x) .>= gamma - 1e-9)
        x_nonzero = x[nonzero_indices];
        x0 = zeros(T, size(x_nonzero))
        W = [Y; -Y]
        if gamma_active
            L = [W[nonzero_indices, :] ones(T, length(nonzero_indices))]
        else
            L = [W[nonzero_indices, :] zeros(T, length(nonzero_indices))]
        end
        F = qr(L, Val(true))
        R = zeros(T, size(F.R) .+ 1)
        R[1:end-1, 1:end-1] = F.R

        new{T, Tf}(x, x_nonzero, x0,
            S, [Y; -Y], gamma,
            H, L, F, R,
            nonzero_indices, zero_indices, gamma_active,
            zeros(T, 0, 0),
            zeros(T, 0), zero(T), zero(T), zeros(T, 0), # Multipliers
            T(tolerance), verbosity, printing_interval,  # Options
            false, 0, NaN, '-', 0, 0,  # Logging
            # Timings
            DataFrame(projection=zeros(T, 0), gradient_steps=zeros(T, 0),
                trs=zeros(T, 0), trs_local=zeros(T, 0),
                move_to_optimizer=zeros(T, 0), move_to_local_optimizer=zeros(T, 0),
                curvature_step=zeros(T, 0), kkt=zeros(T, 0),
                add_constraint=zeros(T, 0), remove_constraint=zeros(T, 0))
        )
    end
end

function remove_constraint!(data, idx::Int)
    if idx > 0
        constraint_idx = data.zero_indices[idx]
        append!(data.nonzero_indices, constraint_idx)
        add_column!(data.H_nonzero, constraint_idx)
        deleteat!(data.zero_indices, idx)
    end
    update_factorizations!(data)
end

function add_constraint!(data, idx::Int)
    if idx <= length(data.nonzero_indices)
        constraint_idx = data.nonzero_indices[idx]
        # print(" | ", constraint_idx, " | ")
        deleteat!(data.nonzero_indices, idx)
        append!(data.zero_indices, constraint_idx)
        remove_column!(data.H_nonzero, idx)
    else
        data.gamma_active = true
    end
    update_factorizations!(data)
end

function solve_sparse_pca(S, gamma::T, y_init::Vector{T}, H=nothing; max_iter=Inf, kwargs...) where T
    if H == nothing
        data = Data(S, gamma, y_init; kwargs...)
    else
        data = Data(S, H, gamma, y_init; kwargs...)
    end

    if data.verbosity > 0
        print_header(data)
        print_info(data)
    end

    while !data.done && data.iteration <= max_iter
        iterate!(data)

        if data.verbosity > 0
            mod(data.iteration, 10*data.printing_interval) == 0 && print_header(data)
            (mod(data.iteration, data.printing_interval) == 0 || data.done) && print_info(data)
        end
    end

    data.verbosity > 0 && println("2-norm lagrange multiplier: ", data.μ)
    data.verbosity > 0 && println("Complementarity: ", dot(data.x[1:size(S, 1)], data.x[size(S, 1)+1:end]))

    return data
end

function iterate_interior!(data::Data{T}) where{T}
    @assert data.gamma_active # This can be achieved by scaling and it keeps things simple
    direction = project!(data, -grad(data, data.x_nonzero)) # Projected gradient
    stepsizes = -data.x_nonzero./min.(direction, 0)
    stepsizes[stepsizes .< 0] .= Inf
    idx = argmin(stepsizes)
    if !isfinite(stepsizes[idx]) || norm(data.x_nonzero + stepsizes[idx]*direction) >= 1
        stepsizes = roots(Poly(
            [norm(data.x_nonzero)^2 - 1,
            2*dot(direction, data.x_nonzero),
            norm(direction)^2]
        ))
        data.x[data.nonzero_indices] = data.x_nonzero + maximum(stepsizes)*direction
    else
        data.x[data.nonzero_indices] = data.x_nonzero + stepsizes[idx]*direction
        add_constraint!(data, idx)
    end
end

function update_factorizations!(data::Data{T}) where{T}
    if data.gamma_active
        data.L = [data.W[data.nonzero_indices, :] ones(T, length(data.nonzero_indices))]
    else
        data.L = [data.W[data.nonzero_indices, :] zeros(T, length(data.nonzero_indices))]
    end
    data.F = qr(data.L, Val(true))
    data.R = zeros(T, size(data.F.R) .+ 1)
    data.R[1:end-1, 1:end-1] = data.F.R

end

function iterate!(data::Data{T}) where{T}
    t_remove, t_add, t_proj, t_grad, t_kkt = zero(T), zero(T), zero(T), zero(T), zero(T)
    t_trs, t_move, t_trs_l, t_move_l, t_curv = zero(T), zero(T), zero(T), zero(T), zero(T)
    data.iteration += 1

    data.x[data.zero_indices] .= 0
    # @show dot(data.x_nonzero, data.H_nonzero.H*data.x_nonzero)/2
    if norm(data.x_nonzero) <= 1 - 1e-9
        @show @elapsed iterate_interior!(data)
        data.x_nonzero = data.x[data.nonzero_indices]
        @elapsed data.x0 = data.x_nonzero - project(data, data.x_nonzero)
        return
    end

    if sqrt(1.0 - norm(data.x0)^2) <= 1e-10
        @warn "LICQ failed. Terminating"
        data.done = true
        return
    end

    t_grad = @elapsed new_constraint = gradient_steps(data, 5)

    projection! = x -> project!(data, x)
    # @assert maximum(data.A*data.x - data.b) <= 1e-8
    if isnan(new_constraint)
        t_trs = @elapsed Xmin, info = trs_boundary(data.H_nonzero.H, zeros(T, length(data.x_nonzero)), one(T), projection!, data.x_nonzero, zero(T), tol=1e-12)
        t_move = @elapsed new_constraint, is_minimizer = move_towards_optimizers!(data, Xmin, info)
        if isnan(new_constraint) && !is_minimizer
            t_trs_l = @elapsed Xmin, info = trs_boundary(data.H_nonzero.H, zeros(T, length(data.x_nonzero)), one(T), projection!, data.x_nonzero, zero(T), tol=1e-12, compute_local=true)
            t_move_l = @elapsed new_constraint, is_minimizer = move_towards_optimizers!(data, Xmin, info)
            if isnan(new_constraint) && !is_minimizer
                t_grad += @elapsed new_constraint = gradient_steps(data)
                if isnan(new_constraint)
                    t_curv = @elapsed new_constraint = curvature_step(data, Xmin[:, 1], info)
                end
            end
        end
    end
    # @show norm(data.x[data.nonzero_indices] - data.x_nonzero)
    data.x .= 0
    data.x[data.nonzero_indices] .= data.x_nonzero

    if !isnan(new_constraint)
        t_add = @elapsed add_constraint!(data, new_constraint)
    else 
        t_kkt = @elapsed removal_idx = check_kkt!(data)
        t_remove = @elapsed remove_constraint!(data, removal_idx)
    end
    data.x_nonzero = data.x[data.nonzero_indices]
    @elapsed data.x0 = data.x_nonzero - project(data, data.x_nonzero)
    
    # Updating, Saving & Printing of timings 
    push!(data.timings, [t_proj, t_grad, t_trs, t_trs_l, t_move, t_move_l, t_curv, t_kkt, t_add, t_remove])
    if data.done && data.verbosity > 1
        data.timings |> CSV.write(string("timings.csv"))
        sums =  [sum(data.timings[i]) for i in 1 : size(data.timings, 2)]
        labels =  names(data.timings)
        show(stdout, "text/plain", [labels sums]); println()
        @printf "Total time (excluding factorizations): %.4e seconds.\n" sum(sums[1:end-2])
    end
    # @assert maximum(data.A*data.x - data.b) <= 1e-8
end

function check_kkt!(data)
    residual_grad = grad(data, data.x_nonzero) + data.μ*data.x_nonzero
    l = data.F\(-residual_grad)
    # show(stdout, "text/plain", data.H_nonzero.H); println();
    # @show eigvals(data.H_nonzero.H), data.μ, l
    data.κ = l[1:end-1]
    data.ν = l[end]
    # println("residual grad norm in nonzeros: ", norm(residual_grad + data.L*l)) # This should be ~zero

    gradient = -sparse_mul(data.S, data.x)
    # n = Int(length(data.x)/2); Sx_ = data.S*(data.x[1:n] - data.x[n+1:end])
    # println("error in gradient calculation:", norm([-Sx_; Sx_] - gradient)) # This should be ~zero
    residual_grad = gradient + data.μ*data.x .+ data.ν + data.W*data.κ
    # @show data.κ
    # @show residual_grad
    data.λ = residual_grad[data.zero_indices]
    residual_grad[data.zero_indices] .-= data.λ

    data.residual = norm(residual_grad)
    tol = -max(1e-8, data.residual)
    removal_idx = 0
    if all(data.λ .>= tol) && data.ν >= tol
        data.done = true
    else
        if minimum(data.λ) <= data.ν
            removal_idx = argmin(data.λ)
        else
            data.gamma_active = false
        end
    end

    return removal_idx
end

function move_towards_optimizers!(data, X, info)
    # @show info
    #= Debugging of the projection accuracy
    for i = 1:size(X, 2) 
        if norm(X[:, i] - (project!(data, X[:, i] - data.x0) + data.x0)) > 1e-11
            @show info
            @show data.gamma_active
            show(stdout, "text/plain", data.L); println()
            @assert false
        end
        # X[:, i] = project!(data, X[:, i] - data.x0) + data.x0
    end
    =#
    if isfeasible(data, X[:, 1])
        data.x_nonzero = X[:, 1]
        data.μ = info.λ[1]
        return NaN, true
    end

    how_many = size(X, 2)
    if how_many == 0
        return NaN
    elseif how_many == 1
        d1 = (data.x_nonzero - data.x0) #; d1 ./= norm(d1)
        d2 = X[:, 1] - data.x_nonzero
        # @show norm(d2 - dot(d1, d2)*d1)
    else
        d1 = data.x_nonzero - X[:, 1]
        d2 = data.x_nonzero - X[:, 2]
    end
    D = qr([d1 d2]).Q*Matrix(I, length(d1), 2)
    new_constraint = minimize_2d(data, D[:, 1], D[:, 2])
    is_minimizer = false
    if isnan(new_constraint)
        if how_many > 1 || norm(data.x_nonzero - X[:, 1]) <= 1e-5
            # @show norm(data.x_nonzero - X[:, 1])
            if how_many == 1 || norm(data.x_nonzero - X[:, 1]) <= norm(data.x_nonzero - X[:, 2])
                data.μ = info.λ[1]
            else
                data.μ = info.λ[2]
            end
            is_minimizer = true
        end
    end
    
    return new_constraint, is_minimizer
end

function gradient_steps(data, max_iter=1e6)
    k = 0;
    new_constraint = NaN
    while isnan(new_constraint) && k < max_iter
        d1 = data.x_nonzero - data.x0;
        d1 ./= norm(d1)
        d2 = project_full(data, grad(data, data.x_nonzero))[1]
        if norm(d2) <= 1e-6
            break
        end
        Q = qr([d1 d2]).Q*Matrix(I, length(d1), 2)

        new_constraint = minimize_2d(data, Q[:, 1], Q[:, 2])
        k += 1
        if mod(k, 500) == 0
            @warn string(k, "th gradient step with f: ", f(data, data.x_nonzero), " proj grad norm: ", norm(d2))
            # @assert k < 1000
        end
    end
    data.grad_steps += k

    return new_constraint
end

function curvature_step(data, x_global::Vector{T}, info) where T
    data.μ = project_full(data, grad(data, data.x_nonzero))[3] # Approximate Lagrange Multiplier
    shift = T(100) # Shift the eigenproblem so that it is easier to solve
    function custom_mul!(y::AbstractVector, x::AbstractVector)
        x .= project_full(data, x)[1]
        mul!(y, data.H_nonzero.H, x)
        axpy!(data.μ - shift, x, y)
        y .= project_full(data, y)[1]
    end
    L = LinearMap{Float64}(custom_mul!, length(x_global); ismutating=true, issymmetric=true)
    λ_min, v_min, _ = eigs(L, nev=1, which=:SR, v0=project_full(data, randn(length(x_global)))[1])
    λ_min = λ_min[1]; v_min = v_min[:, 1]
    λ_min += shift

    if norm(v_min - project_full(data, v_min)[1]) >= 1e-6
        @warn "Search for negative curvature failed; we assume that we are in a local minimum"
        return NaN
    end
    if λ_min >= 0
        @warn "We ended up in an unexpected TRS local minimum. Perhaps the problem is badly scaled." λ_min
        return NaN
    end
    d1 = project!(data, v_min)
    d2 = x_global - data.x_nonzero
    D = qr([d1 d2]).Q*Matrix(I, length(d1), 2)
    new_constraint = minimize_2d(data, D[:, 1], D[:, 2])

    @assert !isnan(new_constraint)
    return new_constraint
end

function find_violations!(violating_constraints::Vector{Bool}, a1::Vector{T}, a2::Vector{T}, b::Vector{T}, x::T, y::T, tol::T) where{T}
    n = length(a1)
    @inbounds for i = 1:n
        violating_constraints[i] = x*a1[i] + y*a2[i] - b[i] >= tol
    end
    return violating_constraints
end

function _minimize_2d(P::Matrix{T}, q::Vector{T}, r::T, a1::Vector{T}, a2::Vector{T}, b::Vector{T}, x0::T, y0::T) where {T}
    grad = P*[x0; y0] + q
    flip = false
    if angle(x0, y0, grad[1], grad[2]) < pi
        flip = true
        # Flip y axis, thus obtaining dot(grad, [x0, y0]) > 0
        y0 = -y0
        a2 = -a2
        P[1, 2] = -P[1, 2]; P[2, 1] = -P[2, 1]
        q[2] = -q[2]
    end

    infeasibility(x, y) = maximum(a1*x + a2*y - b)
    tol = max(1e-12, 1.2*infeasibility(x0, y0))

    new_constraint = NaN
    X, info = trs_boundary_small(P, q, r; compute_local=true)

    if infeasibility(X[1, 1], X[2, 1]) <= tol
        x, y = X[1, 1], X[2, 1]
    else
        θ0 = angle(one(T), zero(T), x0, y0) # atan(y0, x0) 
        δθ_global = angle(x0, y0, X[1, 1], X[2, 1])
        δθ_global = 2*pi - 2e-7 < δθ_global ? zero(T) : δθ_global
        if size(X, 2) == 1
            θ = θ0 + δθ_global
        else
            δθ_local = angle(x0, y0, X[1, 2], X[2, 2])
            δθ_local = 2*pi - 2e-7 < δθ_local ? zero(T) : δθ_local
            θ = θ0 + min(δθ_global, δθ_local)
        end

        n = length(b)
        violating_constraints = Vector{Bool}(undef, n)
        if !any(find_violations!(violating_constraints, a1, a2, b, r*cos(θ), r*sin(θ), tol))
            x, y = r*cos(θ), r*sin(θ)
        else
            f_2d(x, y) = dot([x; y], P*[x; y])/2 + dot([x; y], q)
            θ_low = θ0; θ_high = θ
            # my_constraints = copy(violating_constraints)
            # Binary Search
            for i = 1:100
                find_violations!(violating_constraints, a1, a2, b, r*cos(θ), r*sin(θ), tol)
                if any(violating_constraints) # && f_2d(r*cos(θ), r*sin(θ)) <= f_2d(x0, y0)
                    θ_high = θ
                    new_constraint = findlast(violating_constraints)
                else
                    θ_low = θ
                end
                θ = θ_low + (θ_high - θ_low)/2
            end

            x1, y1, x2, y2 = circle_line_intersections(a1[new_constraint], a2[new_constraint], b[new_constraint], r)
            if infeasibility(x1, y1) <= tol
                if infeasibility(x2, y2) <= tol && f_2d(x2, y2) <= f_2d(x1, y1)
                    x, y = x2, y2
                else
                    x, y = x1, y1
                end
            elseif infeasibility(x2, y2) <= tol
                x, y = x2, y2
            else
                x, y = r*cos(θ_low), r*sin(θ_low) # Just in case the circle_line_intersections fails
                @warn "No constraint identified in minimize_2d"
            end
        end
    end
    if flip; y = -y; end

    return x, y, new_constraint
end

function minimize_2d(data, d1, d2)
    x0 = dot(data.x_nonzero - data.x0, d1); y0 = dot(data.x_nonzero - data.x0, d2)
    z = data.x_nonzero - x0*d1 - y0*d2
    r = sqrt(x0^2 + y0^2)

    # Calculate 2d cost matrix [P11 P12
    #                           P12 P22]
    # and vector [q1; q2]
    Pd1 = data.H_nonzero.H*d1
    P11 = dot(d1, Pd1)
    q1 = dot(Pd1, z) # + dot(d1, data.q) 

    Pd2 = data.H_nonzero.H*d2
    P22 = dot(d2, Pd2)
    P12 = dot(d1, Pd2)
    q2 = dot(Pd2, z) # + dot(d2, data.q)
    P = [P11 P12; P12 P22]; q = [q1; q2]
    # @show norm(P*[x0; y0] + q)

    if data.gamma_active
        b = copy(z)
        a1 = -d1
        a2 = -d2
    else
        b = [z; data.gamma - sum(z)]
        a1 = [-d1; sum(d1)]
        a2 = [-d2; sum(d2)]
    end

    # Discard perfectly correlated constraints. Be careful with this, especially on debugging.
    idx = a1.^2 + a2.^2 .<= 1e-16
    a1[idx] .= 1; a2[idx] .= 0; b[idx] .= 2*r

    x, y, new_constraint = _minimize_2d(P, q, r, a1, a2, b, x0, y0)
    data.x_nonzero = z + x*d1 + y*d2

    return new_constraint
end

function find_closest_minimizer(y1, y2, y_g, y_l)
    theta_g = angle(y1, y2, y_g[1], y_g[2])
    if isempty(y_l)
        return theta_g
    else
        theta_l = angle(y1, y2, y_l[1], y_l[2])
        min(theta_l, theta_g)
    end
end

function angle(x1, y1, x2, y2)
    dot = x1*x2 + y1*y2      # dot product between [x1, y1] and [x2, y2]
    det = x1*y2 - y1*x2      # determinant
    angle = atan(det, dot)  # atan2(y, x) or atan2(sin, cos))
    if angle < 0
        angle += 2*pi
    end
    return angle
end

function isfeasible(data::Data{T}, x) where{T}
    tol = 1.5*max(-minimum(data.x_nonzero), sum(data.x_nonzero) - data.gamma, maximum(abs.(data.L'*(data.x_nonzero - data.x0))), data.tolerance)
    return max(-minimum(x), sum(x) - data.gamma, maximum(abs.(data.L'*(x - data.x0)))) <= tol
end

function f(data::Data{T}, x) where{T}
    return dot(x, data.H_nonzero.H*x)/2
end

function grad(data::Data{T}, x) where{T}
    return data.H_nonzero.H*x
end

function project(data, x)
    return project!(data, copy(x))
end

function project!(data, x)
    return project!(x, data.L, data.F)
end

function project!(x, L::Matrix{T}, F::Factorization{T}) where {T}
    l = F\x
    x .-= L*l
    # @assert norm(data.L'*x) <= 1e-10 norm(data.L'*x)
    return x
end

function project_full(data, x)
    k = findlast(abs.(diag(data.F.R)) .> 1e-11)
    if k == nothing
        return project(data, x), 0, 0
    end
    y = data.x_nonzero - data.x0 # New column to be "added" in the qr factorization
    Qy = data.F.Q'*y
    Qy1 = view(Qy, 1:k)
    Qy2 = view(Qy, k+1:length(y))
    F = qr(Qy2)
    R = view(data.R, 1:k+1, 1:k+1)
    R[1:end-1, end] = Qy1
    R[end, end] = F.factors[1, 1]

    Qx = data.F.Q'*x
    Qx2 = view(Qx, k+1:length(x))
    lmul!(F.Q', Qx2)
    l = UpperTriangular(R)\view(Qx, 1:k+1)
    λ = zeros(size(data.F, 2))
    λ[data.F.p[1:k]] = -l[1:end-1]
    μ = -l[end]
    x_proj = x + data.L*λ + y*μ
    # @show norm(data.L'*x_proj)
    return x_proj, λ, μ
end

#=
For profiling
function iterate!(data::Data{T}) where{T}
    _iterate!(data)
    #=
    Profile.clear()		
    Profile.@profile for i = 1:10; _iterate!(data); end
    Profile.clear()		
    Profile.@profile for i = 1:10; _iterate!(data); end
    ProfileView.view()		
    println("Press enter to continue...")		
    readline(stdin)	
    =#
end
=#