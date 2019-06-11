using LinearAlgebra
using Arpack
using LinearMaps

mutable struct Data{T}
    """
    Data structure for the solution of
        min         ½x'Px + q'x
        subject to  Ax ≤ b
                    ‖x‖ = r
    with an active set algorithm.

    The most important element is F, which holds
    - F.QR:  an updatable QR factorization of the working constraints
    - F.Z:   a view on an orthonormal matrix spanning the nullspace of the working constraints
    - F.P:   the hessian of the problem
    - F.ZPZ: equal to F.Z'*F.P*F.P, i.e. the reduced hessian

    The Data structure also keeps matrices of the constraints not in the working set (A_ignored and b_ignored)
    stored in a continuous manner. This allows efficient use of BLAS.
    """
    x::Vector{T}  # Variable of the minimization problem
    Xmin::Matrix{T}
    y::Vector{T}  # Reduced variable on the nullspace of the working constraints
    x0::Vector{T} # x - F.Z'*y
    f0::T # Constant term in reduced TRS

    n::Int  # length(x)
    m::Int  # Number of constraints

    F::NullspaceHessian{T}  # See first comments on the definition of Data
    q::Vector{T}
    Zq::Vector{T} # Reduced linear cost
    A::Matrix{T}
    b::Vector{T}
    r::T
    working_set::Vector{Int}
    ignored_set::Vector{Int}

    done::Bool
    removal_idx::Int

    A_ignored::SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int}, Base.Slice{Base.OneTo{Int}}}, false}
    b_ignored::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    A_shuffled::Matrix{T}  # That's where A_ignored "views into"
    b_shuffled::Vector{T}  # That's where b_ignored "views into"

    # Options
    tolerance::T
    verbosity::Int
    printing_interval::Int

    # Logging
    iteration::Int
    λ::Vector{T}
    μ::T
    residual::T
    trs_choice::Char
    grad_steps::Int
    eigen_steps::Int
    timings::DataFrame

    function Data(P::Matrix{T}, q::Vector{T}, A::Matrix{T}, b::Vector{T},
        r::T, x::Vector{T}; kwargs...) where T

        m, n = size(A)
        working_set = findall((A*x - b)[:] .>= 1)
        if length(working_set) >= n
            working_set = working_set[1:n-1]
        end
        ignored_set = setdiff(1:m, working_set)

        QR = UpdatableQR(A[working_set, :]')
        A_shuffled = zeros(T, m, n)
        l = length(ignored_set)
        A_shuffled[end-l+1:end, :] .= view(A, ignored_set, :)
        b_shuffled = zeros(T, m)
        b_shuffled[end-l+1:end] .= view(b, ignored_set)

        Data(P, q, A, b, r, x,
            QR, working_set, ignored_set,
            A_shuffled, b_shuffled; kwargs...)
    end

    function Data(P::Matrix{T}, q::Vector{T},  A::Matrix{T}, b::Vector{T}, r::T, x::Vector{T},
        QR::UpdatableQR{T}, working_set::Vector{Int}, ignored_set::Vector{Int},
        A_shuffled::Matrix{T}, b_shuffled::Vector{T};
        verbosity=1, printing_interval=50, tolerance=1e-11) where T

        F = NullspaceHessian{T}(P, QR)
        l = length(ignored_set)

        m, n = size(A)
        λ = zeros(T, m);

        new{T}(x, zeros(T, 0, 0), zeros(T, 0), zeros(T, 0), 0.0,
            n, m, F, q, zeros(T, 0), A, b, r, working_set, ignored_set, false, 0,
            view(A_shuffled, m-l+1:m, :),
            view(b_shuffled, m-l+1:m),
            A_shuffled, b_shuffled,
            T(tolerance), verbosity, printing_interval,  # Options
            0, λ, 0, NaN, '-', 0, 0,  # Logging
            # Timings
            DataFrame(projection=zeros(T, 0), gradient_steps=zeros(T, 0),
            trs=zeros(T, 0), trs_local=zeros(T, 0),
            move_to_optimizer=zeros(T, 0), move_to_local_optimizer=zeros(T, 0),
            curvature_step=zeros(T, 0), kkt=zeros(T, 0),
            add_constraint=zeros(T, 0), remove_constraint=zeros(T, 0))
        )
    end
end

function solve_boundary(P::Matrix{Float64}, q::Vector{Float64}, A::Matrix{Float64}, b::Vector{Float64}, r::Float64,
    x::Vector{Float64}; max_iter=Inf, kwargs...)
    data = Data(P, q, A, b, r, x; kwargs...)

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
    return data.x, [data.λ; data.μ]
end

function iterate!(data::Data{T}) where{T}
    t_remove, t_add, t_proj, t_grad, t_kkt = zero(T), zero(T), zero(T), zero(T), zero(T)
    t_trs, t_move, t_trs_l, t_move_l, t_curv = zero(T), zero(T), zero(T), zero(T), zero(T)

    if data.removal_idx > 0
        t_remove = @elapsed remove_constraint!(data, data.removal_idx)
        data.removal_idx = 0
    end

    t_proj = @elapsed data.y = data.F.Z'*data.x  # Reduced free variable
    if norm(data.y) <= 1e-10
        @warn "LICQ failed. Terminating"
        data.done = true
        return
    end
    data.x0 = data.x - data.F.Z*data.y
    data.Zq = data.F.Z'*(data.q + data.F.P*data.x0) # Reduced q
    data.f0 = dot(data.x0, (data.F.P*data.x0))/2 + dot(data.q, data.x0)

    t_grad = @elapsed new_constraint = gradient_steps(data, 5)
    if isnan(new_constraint) && norm(projected_gradient(data, data.y)) < 1e-9 && length(data.Xmin) > 0
        Ymin = data.F.Z'*data.Xmin # Minimizer of the previous working set
        t_curv += @elapsed new_constraint = curvature_step(data, Ymin[:, 1], false)
    end

    if isnan(new_constraint)
        t_trs = @elapsed Ymin, info = trs_robust(data.F.ZPZ, data.Zq, norm(data.y), tol=1e-11)
        data.Xmin = data.x0 .+ data.F.Z*Ymin;
        t_move = @elapsed new_constraint, is_minimizer = move_towards_optimizers!(data, Ymin)
        if isnan(new_constraint) && !is_minimizer
            t_trs_l = @elapsed Ymin, info = trs_robust(data.F.ZPZ, data.Zq, norm(data.y), tol=1e-11, compute_local=true)
            t_move_l = @elapsed new_constraint, is_minimizer = move_towards_optimizers!(data, Ymin)
            if isnan(new_constraint) && !is_minimizer
                t_grad += @elapsed new_constraint = gradient_steps(data)
                if isnan(new_constraint)
                    t_curv += @elapsed new_constraint = curvature_step(data, Ymin[:, 1])
                end
            end
        end
    end
    data.x = data.x0 + data.F.Z*data.y;
    if !isnan(new_constraint)
        t_add = @elapsed add_constraint!(data, new_constraint)
    end
    if isnan(new_constraint) || data.F.m <= 1
        t_kkt = @elapsed data.removal_idx = check_kkt!(data)
    end

    if data.done && data.verbosity >= 2
        push!(data.timings, [t_proj, t_grad, t_trs, t_trs_l, t_move, t_move_l, t_curv, t_kkt, t_add, t_remove])
        data.timings |> CSV.write(string("timings.csv"))
        sums =  [sum(data.timings[i]) for i in 1 : size(data.timings, 2)]
        labels =  names(data.timings)
        show(stdout, "text/plain", [labels sums]); println()
        @printf "Total time (excluding factorizations): %.4e seconds.\n" sum(sums[1:end-2])
    end
    data.iteration += 1
end

function check_kkt!(data)
    F = qr(data.F.Z'*data.x)
    R = view(data.F.QR.R, 1:data.F.QR.m+1, 1:data.F.QR.m+1)
    R[1:end-1, end] = data.F.QR.Q1'*data.x
    R[end, end] = F.factors[1, 1]

    g = data.F.P*data.x + data.q
    Qg = -[data.F.QR.Q1'*g; (F.Q'*(data.F.QR.Q2'*g))[1]]
    multipliers = UpperTriangular(R)\Qg

    λ = multipliers[1:end-1]
    μ = multipliers[end]
    data.residual = norm(projected_gradient(data, data.y))
    # data.residual = norm(g + [data.A[data.working_set, :]' data.x]*[λ; μ])

    data.λ .= 0
    data.λ[data.working_set] .= λ
    data.μ = μ
    if all(data.λ .>= -max(1e-8, data.residual))
        data.done = true
        data.removal_idx = 0
    else
        data.removal_idx = argmin(λ)
    end

    return data.removal_idx
end

function move_towards_optimizers!(data, Y, flag=true)
    how_many = size(Y, 2)
    if isfeasible(data, Y[:, 1])
        data.y = Y[:, 1]
        data.trs_choice = 'g'
        return NaN, true
    end

    if how_many == 1
        data.trs_choice = 'm'
        d1 = data.y
        d2 = data.y - Y[:, 1];
        minimizer_value = f(data, Y[:, 1])
    else
        data.trs_choice = 'm'
        d1 = data.y - Y[:, 1]
        d2 = data.y - Y[:, 2];
        minimizer_value = max(f(data, Y[:, 1]), f(data, Y[:, 2])) 
    end
    D = qr([d1 d2]).Q
    new_constraint = minimize_2d(data, D[:, 1], D[:, 2])
    if isnan(new_constraint)
        is_minimizer = ((norm(projected_gradient(data, data.y)) <= 1e-6)
                        && (f(data, data.y) <= minimizer_value + max(abs(minimizer_value)/100, 1e-7)))
    else
        is_minimizer = false
    end
    
    return new_constraint, is_minimizer
end

function gradient_steps(data, max_iter=1e4)
    k = 0;
    new_constraint = NaN
    project_grad = projected_gradient(data, data.y)
    while isnan(new_constraint) && k < max_iter && norm(project_grad)/max(abs(f(data, data.y)), 10) >= 1e-8
        d1 = data.y/norm(data.y)
        d2 = project_grad/norm(project_grad)
        new_constraint = minimize_2d(data, d1, d2)
        project_grad = projected_gradient(data, data.y)
        k += 1
        if mod(k, 1000) == 0
            @warn string(k, "th gradient step with f: ", f(data, data.y), " proj grad norm: ", norm(project_grad))
        end
    end
    data.grad_steps += k

    return new_constraint
end

function curvature_step(data, y_global, warning=true)
    # Find tangent eigenvector v_min which should correspond to a negative eigenvalue
    project!(x) = axpy!(-dot(x, data.y)/dot(data.y, data.y), data.y, x)
    l = -(data.y'*(data.F.ZPZ*data.y)+data.Zq'*data.y)/dot(data.y, data.y) - 100 # Approximate Lagrange multiplier
    function custom_mul!(y::AbstractVector, x::AbstractVector)
        # print(dot(x, data.y), " ")
        project!(x)
        mul!(y, data.F.ZPZ, x)
        axpy!(l, x, y)
        project!(y)
    end
    L = LinearMap{Float64}(custom_mul!, data.F.m; ismutating=true, issymmetric=true)
    λ_min, v_min, _ = eigs(L, nev=1, which=:SR, v0=project!(randn(data.F.m)), tol=1e-5)
    λ_min = λ_min[1]; v_min = v_min[:, 1]
    λ_min += 100

    if abs(dot(v_min, data.y)) >= 1e-6
        warning && @warn "Search for negative curvature failed; we assume that we are in a local minimum"
        return NaN
    end
    if λ_min >= -1e-8
        warning && @warn "We ended up in an unexpected TRS local minimum. Perhaps the problem is badly scaled." λ_min
        return NaN
    end
    d1 = project!(v_min)
    d2 = y_global - data.y
    D = qr([d1 d2]).Q

    y_copy = copy(data.y)
    new_constraint = minimize_2d(data, D[:, 1], D[:, 2], false)
    if isnan(new_constraint)
        data.y = y_copy
        new_constraint = minimize_2d(data, D[:, 1], -D[:, 2], false)
        warning && isnan(new_constraint) && @warn "Could not activate a new constraint; checking KKT conditions on a saddle point."
    end
    return new_constraint
end

function find_violations!(violating_constraints::Vector{Bool}, a1::Vector{T}, a2::Vector{T}, b::Vector{T}, x::T, y::T, tol::T) where{T}
    n = length(a1)
    @inbounds for i = 1:n
        violating_constraints[i] = x*a1[i] + y*a2[i] - b[i] >= tol
    end
    return violating_constraints
end

function _minimize_2d(P::Matrix{T}, q::Vector{T}, r::T,
    a1::Vector{T}, a2::Vector{T}, b::Vector{T}, x0::T, y0::T,
    allow_flip=true) where {T}

    grad = P*[x0; y0] + q
    flip = false
    if angle(x0, y0, grad[1], grad[2]) < pi && allow_flip
        flip = true
        # Flip y axis, thus obtaining dot(grad, [x0, y0]) > 0
        y0 = -y0
        a2 = -a2
        P[1, 2] = -P[1, 2]; P[2, 1] = -P[2, 1]
        q[2] = -q[2]
    end

    infeasibility(x, y) = maximum(a1*x + a2*y - b)
    tol = max(1e-12, 1.05*infeasibility(x0, y0))

    new_constraint = NaN
    X, info = trs_boundary_small(P, q, r; compute_local=true)

    if infeasibility(X[1, 1], X[2, 1]) <= tol
        x, y = X[1, 1], X[2, 1]
    else
        θ0 = angle(one(T), zero(T), x0, y0) # atan(y0, x0) 
        δθ_global = angle(x0, y0, X[1, 1], X[2, 1])
        # δθ_global = 2*pi - 2e-7 < δθ_global ? zero(T) : δθ_global
        if size(X, 2) == 1
            δθ = δθ_global
        else
            δθ_local = angle(x0, y0, X[1, 2], X[2, 2])
            # δθ_local = 2*pi - 2e-7 < δθ_local ? zero(T) : δθ_local
            δθ = min(δθ_global, δθ_local)
        end
        θ = θ0 + δθ

        n = length(b)
        violating_constraints = Vector{Bool}(undef, n)

        if !any(find_violations!(violating_constraints, a1, a2, b, r*cos(θ), r*sin(θ), tol))
            x, y = r*cos(θ), r*sin(θ)
        else
            f_2d(x, y) = dot([x; y], P*[x; y])/2 + dot([x; y], q)
            θ_low = θ0; θ_high = θ
            # Binary Search
            for i = 1:80
                find_violations!(violating_constraints, a1, a2, b, r*cos(θ), r*sin(θ), tol)
                if any(violating_constraints) || f_2d(r*cos(θ), r*sin(θ)) > f_2d(x0, y0)
                    θ_high = θ
                    new_constraint = findlast(violating_constraints)
                else
                    θ_low = θ
                end
                θ = θ_low + (θ_high - θ_low)/2
            end

            if new_constraint == nothing || isnan(new_constraint)
                x, y = r*cos(θ), r*sin(θ)
                new_constraint = NaN
            else
                tol = 1.2*tol # Lossen up the tolerance more now that we identified the new constraint
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
                    if abs(a1[new_constraint]*x + a2[new_constraint]*y - b[new_constraint]) >= tol
                        new_constraint = NaN
                    end
                end
            end
        end
    end
    if flip; y = -y; end

    return x, y, new_constraint
end

function minimize_2d(data, d1, d2, allow_flip=true)
    x0 = dot(data.y, d1); y0 = dot(data.y, d2)
    z = data.y - d1*x0 - d2*y0
    r = sqrt(x0^2 + y0^2)

    # Calculate 2d cost matrix [P11 P12
    #                           P12 P22]
    # and vector [q1; q2]
    Pd1 = Symmetric(data.F.ZPZ)*d1
    P11 = dot(d1, Pd1)
    q1 = dot(d1, data.Zq) + dot(Pd1, z)

    Pd2 = Symmetric(data.F.ZPZ)*d2
    P22 = dot(d2, Pd2)
    P12 = dot(d1, Pd2)
    q2 = dot(d2, data.Zq) + dot(Pd2, z)
    P = [P11 P12; P12 P22]; q = [q1; q2]

    b = data.b_ignored - data.A_ignored*(data.F.Z*z + data.x0)
    a1 = data.A_ignored*(data.F.Z*d1)
    a2 = data.A_ignored*(data.F.Z*d2)

    # Discard perfectly correlated constraints
    idx = a1.^2 + a2.^2 .<= 1e-18
    a1[idx] .= 1; a2[idx] .= 0; b[idx] .= 2*r

    x, y, new_constraint = _minimize_2d(P, q, r, a1, a2, b, x0, y0, allow_flip)
    @. data.y = z + x*d1 + y*d2

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
    tol = max(1.5*maximum(data.A*data.x - data.b), data.tolerance)
    return all(data.A_ignored*(data.F.Z*x + data.x0) - data.b_ignored .<= data.tolerance)
end

function f(data::Data{T}, x) where{T}
    return dot(x, data.F.ZPZ*x)/2 + dot(x, data.Zq) + data.f0
end

function grad(data::Data{T}, x) where{T}
    return data.F.ZPZ*x + data.Zq
end

function projected_gradient(data::Data{T}, x) where{T}
    if length(data.F.ZPZ) == 1 # This is because data.y is not always updated after adding a constraint
        return 0*x
    end

    gradient = grad(data, x) 
    F = qr([x gradient])
    if minimum(size(F)) > 1
        v = F.Q[:, 2]
        return dot(v, gradient)*v
    else
        return 0*x
    end
end