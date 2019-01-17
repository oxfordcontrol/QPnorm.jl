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
    y::Vector{T}  # Reduced variable on the nullspace of the working constraints
    x0::Vector{T} # x - F.Z'*y

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

    function Data(P::Matrix{T}, q::Vector{T}, A::Matrix{T}, b::Vector{T},
        r::T, x::Vector{T}; kwargs...) where T

        m, n = size(A)
        @show size(A)
        # working_set = findall((A*x - b)[:] .>= -1e-11)
        working_set = findall((A*x - b)[:] .>= 1.0)
        if length(working_set) >= n - 1
            working_set = working_set[1:n-2]
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

        new{T}(x, zeros(T, 0), zeros(T, 0),
            n, m, F, q, zeros(T, 0), A, b, r, working_set, ignored_set, false, 0,
            view(A_shuffled, m-l+1:m, :),
            view(b_shuffled, m-l+1:m),
            A_shuffled, b_shuffled,
            T(tolerance), verbosity, printing_interval,  # Options
            0, λ, 0, NaN, '-', 0, 0  # Logging
        )
    end
end

function solve_boundary(P::Matrix{T}, q::Vector{T}, A::Matrix{T}, b::Vector{T}, r::T,
    x::Vector{T}; max_iter=Inf, kwargs...) where T
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

    return data.x
end

function iterate!(data::Data{T}) where{T}
    if data.removal_idx > 0
        # print("R", data.working_set[data.removal_idx])
        remove_constraint!(data, data.removal_idx)
        data.removal_idx = 0
    end

    data.y = data.F.Z'*data.x  # Reduced free variable
    data.x0 = data.x - data.F.Z*data.y
    data.Zq = data.F.Z'*(data.q + data.F.P*data.x0) # Reduced q
    if norm(data.y) < 1e-9
        @warn "LICQ failed"
        data.done = true
        return
    end

    new_constraint = gradient_steps(data, 5)

    if isnan(new_constraint)
        if any(isnan.(data.F.ZPZ)) || any(.!isfinite.(data.F.ZPZ))
            @assert false
        end

        # x_global, x_local, info = trs_boundary_small(data.F.ZPZ, data.Zq, norm(data.y); compute_local=true, tol_hard=1e-8)
        x_global, x_local, info = trs_boundary_small(data.F.ZPZ, data.Zq, norm(data.y); compute_local=true)
        new_constraint = move_towards_optimizer!(data, x_global)
        if isnan(new_constraint) && norm(data.y - x_global)/norm(data.y) >= 1e-6
            new_constraint = move_towards_optimizer!(data, x_local)
            if isnan(new_constraint) && (isempty(x_local) || norm(data.y - x_local)/norm(data.y) >= 1e-6)
                new_constraint = gradient_steps(data; x_global=x_global, x_local=x_local)
            end
        end
    end
    # @assert f(data, data.x) <= f(data, data.x0 + data.F.Z*data.y) + 1e-10 f(data, data.x) - f(data, data.x0 + data.F.Z*data.y)
    data.x = data.x0 + data.F.Z*data.y;
    if !isnan(new_constraint)
        # print("A", data.ignored_set[new_constraint])
        add_constraint!(data, new_constraint)
    end
    if isnan(new_constraint) || data.F.m <= 1
        data.removal_idx = check_kkt!(data)
    end
    data.iteration += 1
end

function check_kkt!(data)
    F = qr(data.F.Z'*data.x)
    R = view(data.F.QR.R, 1:data.F.QR.m+1, 1:data.F.QR.m+1)
    @show diag(data.F.QR.R1)
    @show data.A
    R[1:end-1, end] = data.F.QR.Q1'*data.x
    R[end, end] = F.factors[1, 1]
    @show norm(data.y)

    g = grad(data, data.x)
    Qg = -[data.F.QR.Q1'*g; (F.Q'*(data.F.QR.Q2'*g))[1]]
    multipliers = UpperTriangular(R)\Qg
    #=
    multipliers = zeros(0)
    try
        multipliers = UpperTriangular(R)\Qg
    catch
        multipliers = -[data.A[data.working_set, :]; data.x']'\g
    end
    =#


    λ = multipliers[1:end-1]
    μ = multipliers[end]
    # g = grad(data, data.y);  g .-= (data.y/norm(data.y))*dot(g, data.y/norm(data.y))
    # data.residual = norm(g)
    data.residual = norm(g + [data.A[data.working_set, :]' data.x]*[λ; μ])

    data.λ .= 0
    data.λ[data.working_set] .= λ
    data.μ = μ
    # @show minimum(data.λ) data.μ
    if all(data.λ .>= -max(1e-8, norm(data.residual)))
        data.done = true
        data.removal_idx = 0
    else
        data.removal_idx = argmin(λ)
    end

    return data.removal_idx
end

function move_towards_optimizer!(data, x_global)
    if isempty(x_global)
        return NaN
    end
    
    if length(data.ignored_set) == 0
        data.y = x_global
        return NaN
    end

    project!(x) = axpy!(-dot(x, data.y)/dot(data.y, data.y), data.y, x)
    d1 = data.y/norm(data.y)
    d2 = project!(data.y - x_global)
    if norm(d2) <= 1e-9
        return NaN
    end
    d2 ./= norm(d2)
    

    return solve_2d(data, d1, d2)
end

function gradient_steps(data, max_iter=Inf; x_global::Vector=zeros(0), x_local::Vector=zeros(0))
    # @show norm(data.y)
    if length(data.ignored_set) == 0
        return NaN
    end

    n = length(data.y)
    if n == 1
        max_iter = 1
    end
    new_constraint = NaN
    k = 0; h = 0
    while isnan(new_constraint) && k < max_iter
        project!(x) = axpy!(-dot(x, data.y)/dot(data.y, data.y), data.y, x)
        g = project!(grad(data, data.y))
        #=
        @show norm(data.y)
        @show norm(g)
        =#
        if norm(g) > 1e-7 || isfinite(max_iter)
            d1 = data.y/norm(data.y)
            if norm(g) <= 1e-8
                return NaN
            end
            d2 = g/norm(g)
        else
            # return NaN
            d1 = data.y/norm(data.y)
            d2 = similar(d1)
            try
                @warn "Minimum eigenvector computation"
                d2 = project!(minimum_tangent_eigenvector(data))
                d2 ./= norm(d2)
                @assert false
            catch e
                if isa(e, AssertionError)
                    @warn "Badly scaled problem!"
                    return NaN
                end
            end
            h += 1
        end
        new_constraint = solve_2d(data, d1, d2)
        k += 1
        if mod(k, 100) == 0
            # @warn "Gradient steps:" k "residual:" norm(g)
            move_towards_optimizer!(data, x_global)
            if isnan(new_constraint) && norm(data.y - x_global)/norm(data.y) <= 1e-6
                return NaN
            end
            move_towards_optimizer!(data, x_local)
            if isnan(new_constraint) && !isempty(x_local) && norm(data.y - x_local)/norm(data.y) <= 1e-6
                return NaN
            end
        end
    end
    data.grad_steps += k - h
    data.eigen_steps = h

    return new_constraint
end

function solve_2d(data, d1, d2)
    @assert all(.!isnan.(d1))
    @assert all(.!isnan.(d2))
    # @show dot(d1, d2)
    y1 = dot(data.y, d1); y2 = dot(data.y, d2)
    x0 = data.y - d1*y1 - d2*y2
    r = sqrt(y1^2 + y2^2)

    f_2d, P_2d, q_2d = generate_2d_cost(data, d1, d2, x0)
    grad = P_2d*[y1; y2] + q_2d
    # @show grad
    if angle(y1, y2, grad[1], grad[2]) <= pi 
        y2 = -y2
        d2 = -d2
        f_2d, P_2d, q_2d = generate_2d_cost(data, d1, d2, x0)
    end
    y01 = y1; y02 = y2;
    find_infeasible!, isfeasible, a1, a2, b1 = generate_2d_feasibility(data, d1, d2, x0, y1, y2)

    if any(isnan.(P_2d)) || any(.!isfinite.(P_2d))
        @show P_2d, q_2d, r
        @assert false
    end
    y_g, y_l, info = trs_boundary_small(P_2d, q_2d, r; compute_local=true, tol_hard=1e-10)
    #=
    @show P_2d, q_2d, r
    @show y_l, y_g
    @show (P_2d*[y1; y2] .+ q_2d)
    @show info
    =#

    new_constraint = NaN
    f0 = f_2d(y1, y2);# f0 += max(data.tolerance, data.tolerance*abs(f0))
    if isfeasible(y_g[1], y_g[2])
        y1 = y_g[1]; y2 = y_g[2]
    elseif !isempty(y_l) && f_2d(y_l[1], y_l[2]) <= f0 && isfeasible(y_l[1], y_l[2])
        y1 = y_l[1]; y2 = y_l[2]
    else
        theta = find_closest_minimizer(y1, y2, y_g, y_l)
        low = 0.0 #ToDo: Theta0
        high = theta
        #=
        @show r*cos(theta), r*sin(theta)
        @show y1, y2
        @show f0 - f_2d(y_l[1], y_l[2])
        @show y_l, y_g
        =#

        violating_constraints = find_infeasible!(r*cos(theta), r*sin(theta))
        if findlast(violating_constraints) != nothing
            # @show findall(violating_constraints)
            # @show data.ignored_set[findall(violating_constraints)]
            new_constraint = findlast(violating_constraints)
        else
            return NaN
        end

        done = false
        k = 0
        # Binary Search
        # violating_constraints = find_infeasible!(r*cos(theta), r*sin(theta))
        while !done
            theta = low + (high - low)/2
            violating_constraints = find_infeasible!(r*cos(theta), r*sin(theta))
            if findlast(violating_constraints) != nothing
                # @show findall(violating_constraints)
                # @show data.ignored_set[findall(violating_constraints)]
                new_constraint = findlast(violating_constraints)
            end
            if k > 60 # sum(violating_constraints) == 1 || k > 60
                done = true
            elseif sum(violating_constraints) == 0
                low = theta
            else
                high = theta
            end
            k += 1
        end
        if isnan(new_constraint) # This can happen with the eigenvector stuff
            @assert false
        end
        # @show theta
        #=
        violating_constraints = find_infeasible!(r*cos(theta), r*sin(theta))
        =#


        y11, y12, y21, y22 = circle_line_intersections(a1[new_constraint], a2[new_constraint], b1[new_constraint], r)
        dd = (a1*r*cos(theta) + a2*r*sin(theta) - b1)
        # @show y11^2 + y12^2 - r^2
        if maximum(a1*y11 + a2*y12 - b1) <= 0 || maximum(a1*y11 + a2*y12 - b1) <= maximum(a1*y21 + a2*y22 - b1)
            if isfeasible(y21, y22)
                if f_2d(y11, y12) <= f_2d(y21, y22)
                    y1 = y11; y2 = y12
                else
                    y1 = y21; y2 = y22
                end
            else
                y1 = y11; y2 = y12
            end
        else
            y1 = y21; y2 = y22
        end
    end
    if isnan(y1) && isnan(y2)
        #=
        @show (a1*r*cos(theta) + a2*r*sin(theta) - b1)[28]
        @show new_constraint
        @show findall(violating_constraints)
        @show a1[new_constraint], a2[new_constraint], b1[new_constraint]
        =#
        new_constraint = NaN
        y1 = r*cos(theta); y2 = r*sin(theta)
    end
    # @show f_2d(y1, y2), maximum(a1*y1+y2*a2 - b1)
    # if f_2d(y1, y2) > f_2d(y01, y02)
    #    @show f_2d(y1, y2) - f_2d(y01, y02) y1 y2 y01 y02
    # end
    # @assert f_2d(y1, y2) <= f_2d(y01, y02) + 1e-11 
    @. data.y = x0 + y1*d1 + y2*d2

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
    r = sqrt(x2^2 + y2^2)
    if angle < 0
        angle += 2*pi
    end
    return angle
end

function generate_2d_cost(data, d1, d2, x0)
    """
    Returns a function handle f_2d(y1, y2) that calculates the cost of points
    on the plane defined by the two directions data.F.Z*d1 + data.F.Z*d2
    and the point data.F.Z*x0 + data.x0, i.e. giving
    f_2d(y1, y2) = f(data, y1*data.F.Z*d1 + y2*data.F.Z*d2 + data.F.Z*x0 + data.x0)
    (recall that data.F.Z is an orthonormal matrix spanning the nullspace of the working set).
    Also, note that x0 is perpendicular to d1, d2.

    The associated cost matrices P_2d and q_2d are also returned so that
    f_2d(y1, y2) = 1/2*[y1 y2]*P_2d*[y1; y2] + [y1 y2]*q_2d
    """
    Pd1 = data.F.ZPZ*d1
    P11 = dot(d1, Pd1)
    q1 = dot(d1, data.Zq) + dot(Pd1, x0)

    Pd2 = data.F.ZPZ*d2
    P22 = dot(d2, Pd2)
    P12 = dot(d1, Pd2)
    q2 = dot(d2, data.Zq) + dot(Pd2, x0)

    @inline f_2d(y1, y2) = (y1^2*P11 + y2^2*P22)/2 + y1*y2*P12 + y1*q1 + y2*q2

    return f_2d, [P11 P12; P12 P22], [q1; q2]
end

function generate_2d_feasibility(data, d1, d2, x0, y1, y2)
    """
    Returns a function handle isfeasible(y1, y2) that calculates the feasibility (true/false)
    of point on the plane defined by the two directions data.F.Z*d1 + data.F.Z*d2
    and the point data.F.Z*x0 + data.x0, i.e. giving
    isfeasible(y1, y2) = isfeasible(data, y1*data.F.Z*d1 + y2*data.F.Z*d2 + data.F.Z*x0 + data.x0)
    (recall that data.F.Z is an orthonormal matrix spanning the nullspace of the working set).
    Also, note that x0 is perpendicular to d1, d2.

    Three vectors a1, a2, b1 are also returned that give:
    isfeasible(y1, y2) = all(y1*a1 + y2*a2 - b1 .< data.tolerance)
    """
    b1 = data.b_ignored - data.A_ignored*(data.F.Z*x0 + data.x0)
    a1 = data.A_ignored*(data.F.Z*d1)
    a2 = data.A_ignored*(data.F.Z*d2)

    idx = sqrt.(a1.^2 + a2.^2) .<= 1e-9
    a1[idx] .= 1
    a2[idx] .= 0.0
    b1[idx] .= 100*sqrt(y1^2 + y2^2)

    if length(b1) > 0
        tol = max(3*maximum(a1*y1 + a2*y2 - b1), data.tolerance)
    else
        tol = data.tolerance
    end

    n = length(b1)
    violating_constraints = Array{Bool}(undef, n)
    function find_infeasible!(y1::T, y2::T) where{T}
        @inbounds for i = 1:n
            violating_constraints[i] = y1*a1[i] + y2*a2[i] - b1[i] >= tol
        end
        return violating_constraints
    end
    isfeasible(y1, y2) = (findfirst(find_infeasible!(y1, y2)) == nothing)

    return find_infeasible!, isfeasible, a1, a2, b1
end

function minimum_tangent_eigenvector(data)
    # @show data.y
    project!(x) = axpy!(-dot(x, data.y)/dot(data.y, data.y), data.y, x)
    l = -(data.y'*(data.F.ZPZ*data.y)+data.Zq'*data.y)/dot(data.y, data.y) - 100.0 # Approximate Lagrange multiplier
    function custom_mul!(y::AbstractVector, x::AbstractVector)
        @assert(abs(dot(x, data.y)) < 1e-9)
        project!(x)
        mul!(y, data.F.ZPZ, x)
        axpy!(l, x, y)
        project!(y)
    end
    L = LinearMap{Float64}(custom_mul!, data.F.m; ismutating=true, issymmetric=true)
    (λ_min, v_min, nconv, niter, nmult, resid) = eigs(L, nev=1, which=:SR, v0=project!(randn(data.F.m)), tol=1e-10)
    λ_min .+= 100.0

    @assert λ_min[1] < 0 "Error: Either the problem is badly scaled or it belongs to the hard case."
    @assert abs(dot(v_min, data.y)) <= 1e-6 "Error: Either the problem is badly scaled or it belongs to the hard case."

    return v_min[:, 1]/norm(v_min)
end

function isfeasible(data::Data{T}, x) where{T}
    tol = max(2*maximum(data.A*data.x - data.b), data.tolerance)
    if length(x) == data.n
        return all(data.A*x - data.b .<= data.tolerance)
    elseif length(x) == data.F.m
        return all(data.A_ignored*(data.F.Z*x + data.x0) - data.b_ignored .<= data.tolerance)
    else
        throw(DimensionMismatch)
    end
end

function f(data::Data{T}, x) where{T}
    if length(x) == data.n
        return dot(x, data.F.P*x)/2 + dot(x, data.q)
    elseif length(x) == data.F.m
        return dot(x, data.F.ZPZ*x)/2 + dot(x, data.Zq)
    else
        throw(DimensionMismatch)
    end
end

function grad(data::Data{T}, x) where{T}
    if length(x) == data.n
        return data.F.P*x + data.q
    elseif length(x) == data.F.m
        return data.F.ZPZ*x + data.Zq
    else
        throw(DimensionMismatch)
    end
end