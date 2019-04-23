function solve(P::Matrix{T}, q::Vector{T}, A::Matrix{T}, b::Vector{T}, x::Vector{T}; r_min::T=zero(T), r_max::T=T(Inf), max_iter=Inf, kwargs...) where T 
    if r_min < 1e-9; r_min = -one(T); end

    boundary_data, interior_data = create_data(P, q, A, b, r_min, r_max, x; kwargs...)

    active_boundary = which_boundary(x, r_min, r_max)
    if active_boundary == :upper
        data = boundary_data
        data.r = r_max
    elseif active_boundary == :lower
        data = boundary_data
        data.r = r_min
    else
        data = interior_data
    end

    if data.verbosity > 0
        print_header(data)
        print_info(data)
    end

    while !data.done && data.iteration <= max_iter
        iterate!(data)
        if (isa(data, Data) && data.done)
            active_boundary = which_boundary(data.x, r_min, r_max)
            if (active_boundary == :upper && data.μ < -1e-8) ||
                (active_boundary == :lower && data.μ > 1e-8)
                # Don't terminate if the Lagrange multiplier of the spherical constraint is negative
                data.done = false
            end
        end

        if data.verbosity > 0
            mod(data.iteration, 10*data.printing_interval) == 0 && print_header(data)
            (mod(data.iteration, data.printing_interval) == 0 || data.done) && print_info(data)
        end

        active_boundary = which_boundary(data.x, r_min, r_max)
        if isa(data, GeneralQP.Data) && active_boundary != :interior
            # The QP solver has hit the constraint ‖x‖ = r_min or r_max
            # Thus we switch to the etrs_boundary algorithm
            update_data!(boundary_data, interior_data)
            data = boundary_data
        elseif isa(data, Data)
            if (active_boundary == :upper && data.μ < 0 && data.μ < minimum(data.λ)) ||
                (active_boundary == :lower && data.μ > 0 && -data.μ < minimum(data.λ))
                # The etrs_boundary algorithm is in a "stationary" point
                # with the most negative langrange multiplier being the
                # one for the constraint ‖x‖ = r_max or r_min
                # Thus we switch to the GeneralQP solver
                update_data!(interior_data, boundary_data)
                data = interior_data
            end
        end
    end

    if isa(data, Data)
        return data.x, [data.λ; data.μ]
    else
        return data.x, [data.λ; zero(T)]
    end
end

function create_data(P, q, A, b, r_min, r_max, x_init; kwargs...)
    interior_data = GeneralQP.Data(P, q, A, b, x_init; kwargs..., r_max=r_max, r_min=r_min)
    # interior_data.verbosity = 0
    boundary_data = Data(P, q, A, b, r_max, x_init,
        interior_data.F.QR, interior_data.working_set, interior_data.ignored_set,
        interior_data.A_shuffled, interior_data.b_shuffled; kwargs...)
    return boundary_data, interior_data
end

function update_data!(qp::GeneralQP.Data, etrs::Data)
    qp.iteration = etrs.iteration
    if qp.verbosity > 0
        @info "Changing to QP solver."
    end
    # We only have to recompute qp.F.LDL
    # Adding a (temporarily) constraint parallel to x ensures that the
    # projected hessian is p.d. if a shift from r_max to the interior happens
    add_constraint!(etrs.F, etrs.x/norm(etrs.x))
    qp.x = etrs.x
    qp.F.m = qp.F.QR.n - qp.F.QR.m
    qp.F.artificial_constraints = 0  # No need for them now :)
    GeneralQP.update_views!(qp.F)
    try
        qp.F.U.data .= cholesky!(Symmetric(etrs.F.ZPZ[qp.F.m:-1:1, qp.F.m:-1:1])).U
        qp.F.d .= 1
        # remove the temporal constraint parallel to x
        remove_constraint!(qp.F, qp.F.QR.m)
    catch error
        if isa(error, PosDefException)
            # This can only happen during a shift from r_min to the interior
            if qp.verbosity > 0
                @info "Introducing artifical constraints in QP's working set"
            end
            qp.F = NullspaceHessianLDL(qp.F.P, Matrix(view(qp.A, qp.working_set, :)'))
            etrs.F.QR = qp.F.QR # Link the qr of the qp solver and the constant norm solver
            if qp.F.m == 0 # To many artificial constraints...
                remove_constraint!(qp.F, 0)
            end
        else
            throw(error)
        end
    end
    GeneralQP.update_views!(qp)
    if qp.verbosity > 0
        print_header(qp)
        print_info(qp)
    end
end

function update_data!(etrs::Data, qp::GeneralQP.Data)
    if etrs.verbosity > 0
        @info "Changing to constant-norm QP solver."
    end
    etrs.iteration = qp.iteration
    etrs.x = qp.x
    etrs.r = norm(etrs.x)
    # We only have to recompute etrs.F.ZPZ
    etrs.F.m = etrs.F.QR.n - etrs.F.QR.m
    GeneralQP.update_views!(etrs.F)
    etrs.F.ZPZ .= etrs.F.Z'*etrs.F.P*etrs.F.Z
    etrs.F.ZPZ .= (etrs.F.ZPZ + etrs.F.ZPZ')/2
    if etrs.F.m <= 1
        etrs.y = etrs.F.Z'*etrs.x
        etrs.x0 = etrs.x - etrs.F.Z*etrs.y
        etrs.Zq = etrs.F.Z'*(etrs.q + etrs.F.P*etrs.x0) # Reduced q
        if norm(etrs.y) <= 1e-10
            @warn "LICQ failed. Terminating"
            etrs.done = true
        else
            idx = check_kkt!(etrs)
            !etrs.done && idx <= length(etrs.working_set) && remove_constraint!(etrs, idx)
        end
    end
    GeneralQP.update_views!(etrs)
    if etrs.verbosity > 0
        print_header(etrs)
        print_info(etrs)
    end
end

function iterate!(data::GeneralQP.Data{T}) where T
    GeneralQP.iterate!(data)
end

function print_info(data::GeneralQP.Data{T}) where T
    GeneralQP.print_info(data)
end

function print_header(data::GeneralQP.Data{T}) where T
    GeneralQP.print_header(data)
end

function which_boundary(x, r_min, r_max)
    tol = 1e-8
    if norm(x) - r_max >= -tol
        @assert norm(x) - r_max <= tol "Iterate infeasible for upper norm constraint"
        return :upper
    elseif norm(x) - r_min <= tol
        @assert norm(x) - r_min >= -tol "Iterate infeasible for lower norm constraint"
        return :lower
    else
        return :interior
    end
end