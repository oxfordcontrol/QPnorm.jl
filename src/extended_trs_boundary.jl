# Remaining things for a fast, descent implementation
# Rewrite minimum_tangent_eigenvector to project into nullspace of A_working (currently it's ignored)
# Check if the projected_grad < 1e-6 termination criterion in gradient_steps is correct
# Reuse factorizations via the schur complement procedure described in Gould & Toint 2002
# Avoid backslash in check_kkt

using LinearAlgebra
using Arpack
using LinearMaps
using Polynomials

mutable struct Data{T}
    """
    Data structure for the solution of
        min         ½x'Px + q'x
        subject to  Ax ≤ b
                    ‖x‖ = r
    with an active set algorithm.
    """
    x::Vector{T}  # Variable of the minimization problem
    y::Vector{T}
    x0::Vector{T}

    n::Int  # length(x)
    m::Int  # Number of constraints

    P::AbstractMatrix{T}
    q::Vector{T}
    A::AbstractMatrix{T}
    b::Vector{T}
    r::T
    working_set::Vector{Int}
    ignored_set::Vector{Int}

    done::Bool
    removal_idx::Int

    eig_max::T
    ps::MKLPardisoSolver
    M

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

    function Data(P::AbstractMatrix{T}, q::Vector{T}, A::AbstractMatrix{T}, b::Vector{T},
        r::T, x::Vector{T}; kwargs...) where T

        m, n = size(A)
        working_set = findall((A*x - b)[:] .>= -1e-11)
        if length(working_set) >= n
            working_set = working_set[1:n-1]
        end
        ignored_set = setdiff(1:m, working_set)

        Data(P, q, A, b, r, x, working_set, ignored_set; kwargs...)
    end

    function Data(P::AbstractMatrix{T}, q::Vector{T},  A::AbstractMatrix{T}, b::Vector{T}, r::T, x::Vector{T},
        working_set::Vector{Int}, ignored_set::Vector{Int};
        verbosity=1, printing_interval=50, tolerance=1e-11) where T

        m, n = size(A)
        λ = zeros(T, m);
        eig_max = max(maximum(eigs(P, nev=1, which=:LR)[1]), 0)
        @show eig_max

        new{T}(x, zeros(T, 0), zeros(T, 0), n, m, P, q, A, b, r, working_set, ignored_set, false, 0,
            eig_max, MKLPardisoSolver(), nothing,
            T(tolerance), verbosity, printing_interval,  # Options
            0, λ, 0, NaN, '-', 0, 0  # Logging
        )
    end
end

function nullspace_projector!(data, x)
    if length(data.working_set) == 0
        return x
    end
    right_hand_side = [x; zeros(length(data.working_set))]
    left_hand_side = similar(right_hand_side)
    set_phase!(data.ps, 33) # Solve, iterative refinement
    pardiso(data.ps, left_hand_side, data.M, right_hand_side)
    # solve!(data.ps, left_hand_side, data.M, right_hand_side)
    copyto!(x, view(left_hand_side, 1:length(x)))
    #=
    if norm(data.A[data.working_set, :]*x) > 1e-10
        @warn "Inaccurate projection:" norm(data.A[data.working_set, :]*x)
    end
    =#
    return x 
end

function solve_boundary(P::AbstractMatrix{T}, q::Vector{T}, A::AbstractMatrix{T}, b::Vector{T}, r::T,
    x::Vector{T}; kwargs...) where T
    data = Data(P, q, A, b, r, x; kwargs...)

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
    if data.removal_idx > 0
        # remove_constraint!(data, data.removal_idx)
        constraint_idx = data.working_set[data.removal_idx]
        prepend!(data.ignored_set, constraint_idx)
        deleteat!(data.working_set, data.removal_idx)
        data.removal_idx = 0
    end

    data.ps = MKLPardisoSolver()
    if length(data.working_set) > 0
        A_active = data.A[data.working_set, :]
        # R = qr(Matrix(A_active')).R
        # @show minimum(abs.(diag(R))), maximum(abs.(diag(R)))
        
        data.M = [[I A_active']; [A_active -1e-90*I]] # Pardiso all of the diagonal stored
        # Parameters to handle saddle point systems. Some of these are mentioned in the Pardiso Manual.
        set_iparm!(data.ps, 1, 1) # Enable parameters
        set_iparm!(data.ps, 11, 1) # Scaling
        set_iparm!(data.ps, 13, 1) # weighted matchings
        set_iparm!(data.ps, 8, 50) # Max number of iterative refinement steps. MAYBE this is too big?
        set_matrixtype!(data.ps, Pardiso.REAL_SYM_INDEF) # real and symmetric
        set_phase!(data.ps, Pardiso.ANALYSIS_NUM_FACT) # Analysis, numerical factorization
        data.M = get_matrix(data.ps, data.M, :T)
        # set_msglvl!(data.ps, 1)
        # set_iparm!(data.ps, 27, 1) # Check matrices
        right_hand = [data.x; zeros(length(data.working_set))]
        pardiso(data.ps, similar(right_hand), data.M, right_hand)
    end

    @assert abs(norm(data.x) - data.r) < 1e-8
    #=
    d = nullspace_projector!(data, randn(data.n)) # Find a direction in the nullspace of A
    # Calculate alpha such that ‖x + alpha*d‖ = r
    alpha = roots(Poly([norm(data.x)^2 - data.r^2, 2*d'*data.x, norm(d)^2]))
    idx = argmin(abs.(alpha))
    data.x += alpha[idx]*d # Now ‖x‖ = r
    # @show norm(data.x) - data.r
    =#

    data.x0 = data.x - nullspace_projector!(data, copy(data.x))
    # @assert maximum(data.A*data.x - data.b) <= 1e-10
    new_constraint = gradient_steps(data, 5)

    if isnan(new_constraint)
        # ToDo: Change this
        x_global, info = trs_robust(data.P, data.q, norm(data.x), x -> nullspace_projector!(data, x), data.x, data.eig_max; compute_local=false, tol=1e-10, tol_hard=2e-7)
        new_constraint = move_towards_optimizer!(data, x_global)
        if isnan(new_constraint) && norm(data.x - x_global)/norm(data.x) >= 1e-6
            @show "Computing local"
            x_global, x_local, info = trs_robust(data.P, data.q, norm(data.x), x -> nullspace_projector!(data, x), data.x, data.eig_max; compute_local=true, tol=1e-10, tol_hard=2e-7)
            new_constraint = move_towards_optimizer!(data, x_local)
            if isnan(new_constraint) && (isempty(x_local) || norm(data.x - x_local)/norm(data.x) >= 1e-6)
                if !isempty(x_local)
                    @show norm(data.x - x_local)/norm(data.x)
                end
                new_constraint = gradient_steps(data)
            else
                data.μ = info.λ[2]
            end
        else
            data.μ = info.λ[1]
        end
    end
    # @assert norm(data.A[data.working_set, :]*data.x) <= 1e-10
    if !isnan(new_constraint) && new_constraint > 0
        # add_constraint!(data, new_constraint)
        constraint_idx = data.ignored_set[new_constraint]
        deleteat!(data.ignored_set, new_constraint)
        append!(data.working_set, constraint_idx)
    end
    if isnan(new_constraint) || length(data.ignored_set) <= 1 || new_constraint < 0
        data.removal_idx = check_kkt!(data)
    end

    #=
    data.x = constraints_projector!(data, data.x)
    d = nullspace_projector!(data, randn(data.n))
    alpha = roots(Poly([norm(data.x)^2 - data.r^2, 2*d'*data.x, norm(d)^2]))
    data.x += alpha[1]*d
    =#

    if data.iteration > 0
        set_phase!(data.ps, Pardiso.RELEASE_ALL)
        pardiso(data.ps, similar(right_hand), data.M, right_hand)
    end

    data.iteration += 1
end

function check_kkt!(data)
    g = grad(data, data.x)
    g_ = g + data.x*data.μ
    λ = -SparseMatrixCSC(data.A[data.working_set, :]')\g_

    # g = grad(data, data.x);  g .-= (data.x/norm(data.x))*dot(g, data.x/norm(data.x))
    # data.residual = norm(g)
    # ToDo: Change this?
    data.residual = norm(g + [data.A[data.working_set, :]' data.x]*[λ; data.μ])

    data.λ .= 0
    data.λ[data.working_set] .= λ
    if all(data.λ .>= -1e-9)
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
    d1 = (data.x - data.x0); d1 ./= norm(d1)
    d2 = nullspace_projector!(data, data.x - x_global)
    d2 = d2 - dot(d1, d2)*d1
    d2 ./= norm(d2)

    return solve_2d(data, d1, d2)
end

function gradient_steps(data, max_iter=Inf)
    n = length(data.x)
    if n == 1
        max_iter = 1
    end
    new_constraint = NaN
    k = 0; h = 0
    while isnan(new_constraint) && k < max_iter
        g = nullspace_projector!(data, grad(data, data.x))
        #if norm(g) > 1e-5 || isfinite(max_iter)
            d1 = (data.x - data.x0); d1 ./= norm(d1)
            d2 = g - dot(d1, g)*d1
            my_g = copy(d2)
        if norm(d2) > 1e-6 || isfinite(max_iter)
            d2 ./= norm(d2)
        else
            return -1
            d1 = (data.x - data.x0); d1 ./= norm(d1)
            d2 = minimum_tangent_eigenvector(data)
            h += 1
        end
        new_constraint = solve_2d(data, d1, d2)
        k += 1
        if mod(k, 50) == 0
            @warn "Gradient iteration:" k norm(my_g)
        end
    end
    data.grad_steps += k - h
    data.eigen_steps = h

    return new_constraint
end

function solve_2d(data, d1, d2)
    # @assert abs(dot(data.x0, d1)) < 1e-12
    # @assert abs(dot(data.x0, d2)) < 1e-12
    y1 = dot((data.x - data.x0), d1); y2 = dot((data.x - data.x0), d2)
    x0 = data.x - d1*y1 - d2*y2
    r = sqrt(y1^2 + y2^2)
    # norm(data.x - data.x0)
    f_2d, P_2d, q_2d = generate_2d_cost(data, d1, d2, x0)
    grad = P_2d*[y1; y2] + q_2d
    if angle(y1, y2, grad[1], grad[2]) <= pi 
        y2 = -y2
        d2 = -d2
        f_2d, P_2d, q_2d = generate_2d_cost(data, d1, d2, x0)
    end
    y01 = y1
    y02 = y2

    find_infeasible!, isfeasible, a1, a2, b1 = generate_2d_feasibility(data, d1, d2, x0, y1, y2)

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
        theta = find_closest_minimizer(y1, y2, y_g, y_l)
        low = 0.0
        high = theta

        done = false
        k = 0
        # Binary Search
        while !done
            theta = low + (high - low)/2
            violating_constraints = find_infeasible!(r*cos(theta), r*sin(theta))
            if findlast(violating_constraints) != nothing
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

        y11, y12, y21, y22 = circle_line_intersections(a1[new_constraint], a2[new_constraint], b1[new_constraint], r)
        if isfeasible(y11, y12)
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
    delta = y1*d1 + y2*d2
    data.x += (y1-y01)*d1 + (y2-y02)*d2
    # @show dot(delta, x0)
    # @show dot(delta, data.x0)
    # data.x ./= (norm(data.x)/data.r)

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

function angle(x1, x2, y1, y2)
    dot = x1*x2 + y1*y2      # dot product between [x1, y1] and [x2, y2]
    det = x1*y2 - y1*x2      # determinant
    angle = atan(det, dot)  # atan2(y, x) or atan2(sin, cos))
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
    Pd1 = data.P*d1
    P11 = dot(d1, Pd1)
    q1 = dot(d1, data.q) + dot(Pd1, x0)

    Pd2 = data.P*d2
    P22 = dot(d2, Pd2)
    P12 = dot(d1, Pd2)
    q2 = dot(d2, data.q) + dot(Pd2, x0)

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
    b1 = (data.b - data.A*x0)[data.ignored_set]
    a1 = (data.A*d1)[data.ignored_set]
    a2 = (data.A*d2)[data.ignored_set]

    tol = max(2*maximum(a1*y1 + a2*y2 - b1), data.tolerance)

    violating_constraints = Array{Bool}(undef, length(b1))
    function find_infeasible!(y1::T, y2::T) where{T}
        n = length(b1)
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
    l = -(data.y'*(data.F.ZPZ*data.y)+data.Zq'*data.y)/dot(data.y, data.y) # Approximate Lagrange multiplier
    function custom_mul!(y::AbstractVector, x::AbstractVector)
        mul!(y, data.F.ZPZ, x)
        axpy!(l, x, y)
        project!(y)
    end
    L = LinearMap{Float64}(custom_mul!, data.F.m; ismutating=true, issymmetric=true)
    (λ_min, v_min, nconv, niter, nmult, resid) = eigs(L, nev=1, which=:SR, v0=project!(randn(data.F.m)))

    @show λ_min
    @show abs(dot(v_min, data.y))
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
        return dot(x, data.P*x)/2 + dot(x, data.q)
    else
        throw(DimensionMismatch)
    end
end

function grad(data::Data{T}, x) where{T}
    if length(x) == data.n
        return data.P*x + data.q
    else
        throw(DimensionMismatch)
    end
end