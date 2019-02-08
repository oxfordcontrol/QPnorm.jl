using LinearAlgebra
using Arpack
using LinearMaps
using SparseArrays
using Printf

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
    x0::Vector{T} # x - projection(x) where projection is onto the nullspace of the working constraints

    n::Int  # length(x)
    m::Int  # Number of constraints

    P::SparseMatrixCSC{T}
    q::Vector{T}
    A::SparseMatrixCSC{T}
    b::Vector{T}
    r::T
    working_set::Vector{Int}
    ignored_set::Vector{Int}

    done::Bool
    removal_idx::Int

    Xmin::Array{T} # Last TRS minimizer(s)
    eig_max::T # Maximum eigenvalue of P

    F  # Auxilliary matrix of Pardiso
    ps # Pardiso object

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

    #=
    # Timings
    t_global_trs::Vector{T}
    t_local_trs::Vector{T}
    t_gradient::Vector{T}
    t_add_constraint::Vector{T}
    t_move::Vector{T}
    t_remove_constraint::T
    t_kkt::Vector{T}
    =#

    function Data(P::SparseMatrixCSC{T}, q::Vector{T}, A::SparseMatrixCSC{T}, b::Vector{T},
        r::T, x::Vector{T}; kwargs...) where T

        m, n = size(A)
        working_set = findall((A*x - b)[:] .>= 1)
        if length(working_set) >= n
            working_set = working_set[1:n-1]
        end
        ignored_set = setdiff(1:m, working_set)

        Data(P, q, A, b, r, x, working_set, ignored_set; kwargs...)
    end

    function Data(P::SparseMatrixCSC{T}, q::Vector{T}, A::SparseMatrixCSC{T}, b::Vector{T}, r::T, x::Vector{T},
        working_set::Vector{Int}, ignored_set::Vector{Int};
        verbosity=1, printing_interval=50, tolerance=1e-11) where T

        m, n = size(A)
        λ = zeros(T, m);
        new{T}(x, zeros(T, 0),
            n, m,
            P, q, A, b, r,
            working_set, ignored_set,
            false, 0,
            zeros(T, 0),
            eigs(P, nev=1, which=:LR, ritzvec=false)[1][1],
            nothing, nothing,
            T(tolerance), verbosity, printing_interval,  # Options
            0, λ, 0, NaN, '-', 0, 0,  # Logging
            DataFrame(projection=zeros(T, 0), gradient_steps=zeros(T, 0),
                trs=zeros(T, 0), trs_local=zeros(T, 0),
                move_to_optimizer=zeros(T, 0), move_to_local_optimizer=zeros(T, 0),
                curvature_step=zeros(T, 0), kkt=zeros(T, 0),
                add_constraint=zeros(T, 0), remove_constraint=zeros(T, 0))
        )
    end
end

function fact(data)
    if data.ps == nothing
        data.ps = MKLPardisoSolver()
        set_matrixtype!(data.ps, Pardiso.REAL_SYM_INDEF) # real and symmetric
        pardisoinit(data.ps) # Set default options for the matrix type we have chosen
        # set_nprocs!(data.ps, 4) # Number of threads used by MKL pardiso. Warning: BLAS is recommened to be single threaded.
        set_iparm!(data.ps, 1, 1) # Enable parameters
        # set_iparm!(data.ps, 25, 1) # Parallelize forward/backward solve
        set_iparm!(data.ps, 11, 1) # Scaling
        set_iparm!(data.ps, 13, 1) # weighted matchings
        # set_iparm!(data.ps, 8, 6) # Max number of iterative refinement steps. MAYBE this is too big?
    end
    A_working = SparseMatrixCSC(data.A[data.working_set, :])
    if length(A_working) > 0
        M = SparseMatrixCSC([[I A_working']; [A_working -1e-90*I]]) # Pardiso all of the diagonal stored
    else
        M = SparseMatrixCSC(1.0*I, data.n, data.n)
    end
    data.F = get_matrix(data.ps, M, :N)
    set_phase!(data.ps, Pardiso.ANALYSIS_NUM_FACT) # Analysis, numerical factorization
    pardiso(data.ps, zeros(size(M, 1)), data.F, zeros(size(M, 1)))
end

function remove_constraint!(data, idx::Int)
    constraint_idx = data.working_set[idx]
    prepend!(data.ignored_set, constraint_idx)
    deleteat!(data.working_set, idx)
    fact(data)
end

function add_constraint!(data, idx::Int)
    constraint_idx = data.ignored_set[idx]
    deleteat!(data.ignored_set, idx)
    append!(data.working_set, constraint_idx)
    fact(data)
end

function solve_boundary(P::SparseMatrixCSC{T}, q::Vector{T}, A::SparseMatrixCSC{T}, b::Vector{T}, r::T,
    x::Vector{T}; max_iter=Inf, kwargs...) where T
    data = Data(P, q, A, b, r, x; kwargs...)
    fact(data)

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
    _iterate!(data)
    #=
    Profile.clear()		
    Profile.@profile for i = 1:100; _iterate!(data); end
    Profile.clear()		
    Profile.@profile for i = 1:100; _iterate!(data); end
    ProfileView.view()		
    println("Press enter to continue...")		
    readline(stdin)	
    =#
end

function _iterate!(data::Data{T}) where{T}
    t_remove, t_add, t_proj, t_grad, t_kkt = zero(T), zero(T), zero(T), zero(T), zero(T)
    t_trs, t_move, t_trs_l, t_move_l, t_curv = zero(T), zero(T), zero(T), zero(T), zero(T)

    @assert maximum(data.A*data.x - data.b) <= 1e-10
    if data.removal_idx > 0
        t_remove = @elapsed remove_constraint!(data, data.removal_idx)
        data.removal_idx = 0
    end

    t_proj = @elapsed data.x0 = data.x - project(data, data.x)
    if sqrt(data.r^2 - norm(data.x0)^2) <= 1e-10
        @warn "LICQ failed. Terminating"
        data.done = true
        return
    end

    t_grad = @elapsed new_constraint = gradient_steps(data, 5)

    @assert maximum(data.A*data.x - data.b) <= 1e-10
    if isnan(new_constraint)
        projection! = x -> project!(data, x)
        t_trs = @elapsed Xmin, info = trs_boundary(data.P, data.q, data.r, projection!, data.x, data.eig_max, tol=1e-12)
        #=
        Profile.init(n = 10^8, delay = 1e-5)
        Profile.clear()
        Profile.@profile trs_boundary(data.P, data.q, data.r, projection!, data.x, data.eig_max, tol=1e-12)
        ProfileView.view()		
        println("Press enter to continue...")		
        readline(stdin)	
        =#
        # @show norm((data.A*(Xmin .- data.x0))[data.working_set])
        #=
        q_ = data.F.Z'*(data.q + data.P*data.x0) # Reduced q 
        Ymin, info_ = trs_robust(Symmetric(data.F.ZPZ), q_, r_, tol=1e-11)
        @show info.λ - info_.λ
        =#
        t_move = @elapsed new_constraint, is_minimizer = move_towards_optimizers!(data, Xmin, info)
        if isnan(new_constraint) && !is_minimizer
            t_trs_l = @elapsed Xmin, info = trs_boundary(data.P, data.q, data.r, x -> x .= project(data, x), data.x, data.eig_max, tol=1e-12, compute_local=true)
            t_move_l = @elapsed new_constraint, is_minimizer = move_towards_optimizers!(data, Xmin, info)
            if isnan(new_constraint) && !is_minimizer
                t_grad += @elapsed new_constraint = gradient_steps(data)
                if isnan(new_constraint)
                    t_curv = @elapsed new_constraint = curvature_step(data, Ymin[:, 1], info)
                end
            end
        end
    end
    if !isnan(new_constraint)
        t_add = @elapsed add_constraint!(data, new_constraint)
    end
    if isnan(new_constraint) || data.n - length(data.working_set) <= 1
        t_kkt = @elapsed data.removal_idx = check_kkt!(data)
    end
    
    push!(data.timings, [t_proj, t_grad, t_trs, t_trs_l, t_move, t_move_l, t_curv, t_kkt, t_add, t_remove])
    if data.done
        data.timings |> CSV.write(string("timings.csv"))
        sums =  [sum(data.timings[i]) for i in 1 : size(data.timings, 2)]
        labels =  names(data.timings)
        show(stdout, "text/plain", [labels sums]); println()
        @printf "Total time (excluding factorizations): %.4e seconds.\n" sum(sums[1:end-2])
    end
    @assert maximum(data.A*data.x - data.b) <= 1e-10
    data.iteration += 1
end

function check_kkt!(data)
    g = grad(data, data.x)
    λ = kkt(data, -g - data.μ*data.x)
    # show(stdout, "text/plain", [x_ data.x])
    # @show norm(data.x - x_)
    # @show norm(x_), data.r
    #=
    multipliers = [data.A[data.working_set, :]' data.x]\(-g)
    @show multipliers[end] - data.μ
    show(stdout, "text/plain", [multipliers[1:end-1] λ multipliers[1:end-1]-λ]); println()
    @assert false
    F = qr(data.F.Z'*data.x)
    R = view(data.F.QR.R, 1:data.F.QR.m+1, 1:data.F.QR.m+1)
    R[1:end-1, end] = data.F.QR.Q1'*data.x
    R[end, end] = F.factors[1, 1]

    g = grad(data, data.x)
    Qg = -[data.F.QR.Q1'*g; (F.Q'*(data.F.QR.Q2'*g))[1]]
    multipliers = UpperTriangular(R)\Qg
    =#

    data.λ .= 0
    data.λ[data.working_set] .= λ

    data.residual = norm(g + data.A'*data.λ + data.x*data.μ)
    # data.residual = norm(Qg[length(λ)+1:end])
    if all(data.λ .>= -max(1e-8, data.residual))
        data.done = true
        data.removal_idx = 0
    else
        # @show minimum(λ)
        data.removal_idx = argmin(λ)
    end

    return data.removal_idx
end

function move_towards_optimizers!(data, X, info)
    if isfeasible(data, X[:, 1])
        data.x = X[:, 1]
        data.μ = info.λ[1]
        return NaN, true
    end

    how_many = size(X, 2)
    if how_many == 0
        return NaN
    elseif how_many == 1
        d1 = (data.x - data.x0) #; d1 ./= norm(d1)
        d2 = X[:, 1] - data.x
        # @show norm(d2 - dot(d1, d2)*d1)
    else
        d1 = data.x - X[:, 1]
        d2 = data.x - X[:, 2]
    end
    D = qr([d1 d2]).Q*Matrix(I, length(d1), 2)
    new_constraint = minimize_2d(data, D[:, 1], D[:, 2])
    is_minimizer = false
    if isnan(new_constraint)
        if how_many > 1 || norm(data.x - X[:, 1])/data.r <= 1e-5
            @show norm(data.x - X[:, 1])
            if how_many == 1 || norm(data.x - X[:, 1]) <= norm(data.x - X[:, 2])
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
        d1 = data.x - data.x0;
        d1 ./= norm(d1)
        d2 = project(data, grad(data, data.x))
        # @show norm(d2 - dot(d1, d2)*d1)
        if norm(d2 - dot(d1, d2)*d1) <= 1e-6
            break
        end
        # if norm(project_grad) <= 1e-8 # ToDo relative to objective value
        # end
        Q = qr([d1 d2]).Q*Matrix(I, data.n, 2)
        new_constraint = minimize_2d(data, Q[:, 1], Q[:, 2])
        k += 1
        if mod(k, 500) == 0
            @warn string(k, "th gradient step with f: ", f(data, data.x), " proj grad norm: ", norm(project_grad))
            # @assert k < 1000
        end
    end
    data.grad_steps += k

    return new_constraint
end

function curvature_step(data, y_global)
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
    λ_min, v_min, _ = eigs(L, nev=1, which=:SR, v0=project!(randn(data.F.m)))
    λ_min = λ_min[1]; v_min = v_min[:, 1]
    λ_min += 100

    if abs(dot(v_min, data.y)) >= 1e-6
        @warn "Search for negative curvature failed; we assume that we are in a local minimum"
        return NaN
    end
    if λ_min >= 0
        @warn "We ended up in an unexpected TRS local minimum. Perhaps the problem is badly scaled." λ_min
        return NaN
    end
    d1 = project!(v_min)
    d2 = y_global - data.y
    D = qr([d1 d2]).Q
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
    tol = T(max(1e-11, 1.01*infeasibility(x0, y0)))

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
        violating_constraints = Array{Bool}(undef, n)
        if infeasibility(r*cos(θ), r*sin(θ)) <= tol
            x, y = r*cos(θ), r*sin(θ)
        else
            f_2d(x, y) = dot([x; y], P*[x; y])/2 + dot([x; y], q)
            θ_low = T(θ0); θ_high = T(θ)
            θ = T(θ)
            # Binary Search
            # f0 = f_2d(x0, y0)
            f0 = f_2d(x0, y0); x = NaN; y = NaN; idx = NaN
            for i = 1:80
                find_violations!(violating_constraints, a1, a2, b, r*cos(θ), r*sin(θ), tol)
                if any(violating_constraints)
                    θ_high = θ
                    idx = findlast(violating_constraints) # We're lazy here, we could check all of them
                    x1, y1, x2, y2 = circle_line_intersections(a1[idx], a2[idx], b[idx], r)
                    if !any(find_violations!(violating_constraints, a1, a2, b, x1, y1, tol))
                        x, y = x1, y1
                        new_constraint = idx
                    end
                    if (f_2d(x2, y2) <= f_2d(x1, y2) || isnan(new_constraint)) && !any(find_violations!(violating_constraints, a1, a2, b, x2, y2, tol))
                        x, y = x2, y2
                        new_constraint = idx
                    end
                    if !isnan(new_constraint); break; end
                else
                    θ_low = θ
                end
                θ = θ_low + (θ_high - θ_low)/2
            end
            # @assert !isnan(x) && !isnan(y) && !isnan(new_constraint)
            if isnan(new_constraint)
                @warn "No constraint in minimize 2d"
                # new_constraint = idx
                x, y = r*cos(θ), r*sin(θ)
                new_constraint = find_violations!(violating_constraints, a1, a2, b, x, y, tol)
            end
        end
    end
    if flip; y = -y; end

    return x, y, new_constraint
end

function minimize_2d(data, d1, d2)
    x0 = dot(data.x - data.x0, d1); y0 = dot(data.x - data.x0, d2)
    z = data.x - x0*d1 - y0*d2
    r = sqrt(x0^2 + y0^2)

    # Calculate 2d cost matrix [P11 P12
    #                           P12 P22]
    # and vector [q1; q2]
    Pd1 = data.P*d1
    P11 = dot(d1, Pd1)
    q1 = dot(d1, data.q) + dot(Pd1, z)

    Pd2 = data.P*d2
    P22 = dot(d2, Pd2)
    P12 = dot(d1, Pd2)
    q2 = dot(d2, data.q) + dot(Pd2, z)
    P = [P11 P12; P12 P22]; q = [q1; q2]

    b = (data.b - data.A*z)[data.ignored_set]
    a1 = (data.A*d1)[data.ignored_set]
    a2 = (data.A*d2)[data.ignored_set]

    # Discard perfectly correlated constraints. Be careful with this, especially on debugging.
    idx = a1.^2 + a2.^2 .<= 1e-16
    a1[idx] .= 1; a2[idx] .= 0; b[idx] .= 2*r

    x, y, new_constraint = _minimize_2d(P, q, r, a1, a2, b, x0, y0)
    data.x = z + x*d1 + y*d2

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
    return all(data.A*x - data.b .<= tol)
end

function f(data::Data{T}, x) where{T}
    return dot(x, data.P*x)/2 + dot(x, data.q)
end

function grad(data::Data{T}, x) where{T}
    return data.P*x + data.q
end

function project(data, x)
    z = [x; zeros(length(data.working_set))]
    y = similar(z)
    set_phase!(data.ps, 33) # Solve, iterative refinement
    pardiso(data.ps, y, data.F, z)
    return y[1:data.n]
end

function project!(data::Data{T}, x::AbstractVector{T}) where {T}
    z = [x; zeros(T, length(data.working_set))]
    y = similar(z)
    set_phase!(data.ps, 33) # Solve, iterative refinement
    pardiso(data.ps, y, data.F, z)
    copyto!(x, view(y, 1:data.n))
end

function kkt(data, g)
    z = [g; zeros(length(data.working_set))]
    y = similar(z)
    set_phase!(data.ps, 33) # Solve, iterative refinement
    pardiso(data.ps, y, data.F, z)
    return y[data.n+1:end]
end

#=
function project(data, x)
    return data.F.Z*(data.F.Z'*x)
end
=#

#=
function project(data, x)
    λ = data.A_*x
    λ = data.F_\λ
    return x - data.A_'*λ
end
=#

#=
function project(x)
    F = qr(data.F.Z'*data.x)
    return data.F.Z*(F.Q*(F.Q'*(data.F.Z'*g)))
end
=#