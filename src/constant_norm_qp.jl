using LinearAlgebra
using Arpack
using TRS

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
    idx::Int

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
        r::T, x::Vector{T}; verbosity=1, printing_interval=50, tolerance=1e-10) where T

        m, n = size(A)
        working_set = findall((A*x - b)[:] .>= -1e-11)
        ignored_set = setdiff(1:m, working_set)

        F = NullspaceHessian{T}(P, A[working_set, :]')
        A_shuffled = zeros(T, m, n)
        l = length(ignored_set)
        A_shuffled[end-l+1:end, :] .= view(A, ignored_set, :)
        b_shuffled = zeros(T, m)
        b_shuffled[end-l+1:end] .= view(b, ignored_set)

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

function solve(P::Matrix{T}, q::Vector{T}, A::Matrix{T}, b::Vector{T}, r::T,
    x::Vector{T}; kwargs...) where T
    data = Data(P, q, A, b, r, x)

    if data.verbosity > 0
        print_header(data)
        print_info(data)
    end

    while !data.done && data.iteration <= Inf
        iterate!(data)

        if data.verbosity > 0
            mod(data.iteration, 10*data.printing_interval) == 0 && print_header(data)
            (mod(data.iteration, data.printing_interval) == 0 || data.done) && print_info(data)
        end
    end
    return x
end

function iterate!(data::Data{T}) where{T}
    data.y = data.F.Z'*data.x  # Reduced free variable
    data.x0 = data.x - data.F.Z*data.y
    data.Zq = data.F.Z'*(data.q + data.F.P*data.x0) # Reduced q

    new_constraint = gradient_steps(data, 5)

    if isnan(new_constraint)
        x_global, x_local, info = trs_boundary(data.F.ZPZ, data.Zq, norm(data.y); compute_local=true)
        f0 = f(data, data.y); f0 += max(data.tolerance, data.tolerance*abs(f0))
        if isfeasible(data, x_global)
            data.y = x_global; data.trs_choice = 'g'
        elseif !isempty(x_local) && f(data, x_local) <= f0 && isfeasible(data, x_local)
            data.y = x_local; data.trs_choice = 'l'
        else
            new_constraint = gradient_steps(data)
        end
    end
    data.x = data.x0 + data.F.Z*data.y;
    if !isnan(new_constraint)
        add_constraint!(data, new_constraint)
    end
    if isnan(new_constraint) || data.F.m <= 1
        idx = check_kkt!(data)
        !data.done && idx <= length(data.working_set) && remove_constraint!(data, idx)
    end
    data.iteration += 1
end

function check_kkt!(data)
    F = qr(data.F.Z'*data.x)
    R = view(data.F.QR.R, 1:data.F.QR.m+1, 1:data.F.QR.m+1)
    R[1:end-1, end] = data.F.QR.Q1'*data.x
    R[end, end] = F.factors[1, 1]

    g = grad(data, data.x)
    Qg = -[data.F.QR.Q1'*g; (F.Q'*(data.F.QR.Q2'*g))[1]]
    multipliers = UpperTriangular(R)\Qg

    λ = multipliers[1:end-1]
    μ = multipliers[end]
    # g = grad(data, data.y);  g .-= (data.y/norm(data.y))*dot(g, data.y/norm(data.y))
    # data.residual = norm(g)
    data.residual = norm(g + [data.A[data.working_set, :]' data.x]*[λ; μ])

    idx = NaN
    if all(multipliers .>= 0)
        data.λ .= 0
        data.λ[data.working_set] .= λ
        data.μ = μ
        data.done = true
    else
        data.idx = argmin(multipliers)
    end

    return data.idx
end

function gradient_steps(data, max_iter=Inf)
    n = length(data.y)

    new_constraint = NaN
    k = 0; h = 0
    while isnan(new_constraint) && k < max_iter
        project!(x) = axpy!(-dot(x, data.y)/dot(data.y, data.y), data.y, x)
        g = project!(grad(data, data.y))
        if norm(g) > 1e-5 || isfinite(max_iter)
            d1 = data.y/norm(data.y)
            d2 = g/norm(g)
            if any(isnan.(d2))
                @show norm(project!(grad(data, data.y))), norm(grad(data, data.x0 + data.F.Z*data.y))
                @show k
            end
        else
            d1 = data.y/norm(data.y)
            d2 = minimum_tangent_eigenvector(data)
            h += 1
        end
        new_constraint = solve_2d(data, d1, d2)
        k += 1
    end
    data.grad_steps += k - h
    data.eigen_steps = h

    return new_constraint
end

function solve_2d(data, d1, d2)
    y1 = dot(data.y, d1); y2 = dot(data.y, d2)
    x0 = data.y - d1*y1 - d2*y2
    r = sqrt(y1^2 + y2^2)

    f_2d, P_2d, q_2d = generate_2d_cost(data, d1, d2, x0)
    isfeasible, a1, a2, b1 = generate_2d_feasibility(data, d1, d2, x0)

    if any(isnan.(P_2d)) || any(isnan.(q_2d)) || isnan(r)
        @show P_2d, q_2d, r
    end
    y_g, y_l, info = trs_boundary_small(P_2d, q_2d, r; compute_local=true)

    new_constraint = NaN
    f0 = f_2d(y1, y2); f0 += max(data.tolerance, data.tolerance*abs(f0))
    if isfeasible(y_g[1], y_g[2])
        y1 = y_g[1]; y2 = y_g[2]
    elseif !isempty(y_l) && f_2d(y_l[1], y_l[2]) <= f0 && isfeasible(y_l[1], y_l[2])
        y1 = y_l[1]; y2 = y_l[2]
    else
        min_value = f0 
        for i = 1:length(a1)
            y11, y12, y21, y22 = circle_line_intersections(a1[i], a2[i], b1[i], r)
            if f_2d(y11, y12) <= min_value && isfeasible(y11, y12)
                y1 = y11; y2 = y12;
                new_constraint = i
                min_value = f_2d(y1, y2)
            end
            if f_2d(y21, y22) <= min_value && isfeasible(y21, y22)
                y1 = y21; y2 = y22;
                new_constraint = i
                min_value = f_2d(y1, y2)
            end
        end
        # Solution must activate a new constraint as all of the minimisers are infeasible.
        @assert !isnan(new_constraint) || debug_gradient_step(a1, a2, b1, r, y_l, y_g, f0, f_2d)
    end
    @. data.y = x0 + y1*d1 + y2*d2

    return new_constraint
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

function generate_2d_feasibility(data, d1, d2, x0)
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

    violating_constraints = Array{Bool}(undef, length(b1))
    function find_infeasible!(y1::T, y2::T) where{T}
        n = length(b1)
        @inbounds for i = 1:n
            violating_constraints[i] = y1*a1[i] + y2*a2[i] - b1[i] >= data.tolerance
        end
        return findfirst(violating_constraints)
    end
    isfeasible(y1, y2) = (find_infeasible!(y1, y2) == nothing)

    return isfeasible, a1, a2, b1
end

function minimum_tangent_eigenvector(data)
    project!(x) = axpy!(-dot(x, data.y)/dot(data.y, data.y), data.y, x)
    l = -(data.y'*(data.F.ZPZ*data.y)+data.Zq'*data.y)/dot(data.y, data.y) # Approximate Lagrange multiplier
    function custom_mul!(y::AbstractVector, x::AbstractVector)
        mul!(y, data.F.ZPZ, x)
        axpy!(l, x, y)
        project!(y)
    end
    L = LinearMap{Float64}(custom_mul!, data.F.m; ismutating=true, issymmetric=true)
    (λ_min, v_min, nconv, niter, nmult, resid) = eigs(L, nev=1, which=:SR, v0=project!(randn(data.F.m)))

    @assert λ_min[1] < 0 "Error: Either the problem is badly scaled or it belongs to the hard case."
    @assert abs(dot(v_min, data.y)) <= 1e-6 "Error: Either the problem is badly scaled or it belongs to the hard case."

    return v_min[:, 1]/norm(v_min)
end

function isfeasible(data::Data{T}, x) where{T}
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
        return dot(x, data.P*x)/2 + dot(x, data.q)
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