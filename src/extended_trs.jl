function solve(P::Matrix{T}, q::Vector{T}, A::Matrix{T}, b::Vector{T}, r::T,
    x::Vector{T}; max_iter=Inf, kwargs...) where T 

    boundary_data, interior_data = create_data(P, q, A, b, r, x; kwargs...)

    @assert(norm(x) <= r + 1e-12)
    if norm(x) >= r - 1e-10
        data = boundary_data
    else
        data = interior_data
    end

    if data.verbosity > 0
        print_header(data)
        print_info(data)
    end

    while !data.done && data.iteration <= max_iter
        iterate!(data)
        if isa(data, Data) && data.done && data.μ < -1e-8
            # Don't terminate if the Lagrange multiplier of the spherical constraint is negative
            data.done = false
        end

        if data.verbosity > 0
            mod(data.iteration, 10*data.printing_interval) == 0 && print_header(data)
            (mod(data.iteration, data.printing_interval) == 0 || data.done) && print_info(data)
        end

        if isa(data, GeneralQP.Data) && norm(data.x) >= r - 1e-10
            # The QP solver has hit the constraint ‖x‖ = r
            # Thus we switch to the etrs_boundary algorithm
            update_data!(boundary_data, interior_data)
            data = boundary_data
        elseif isa(data, Data) && data.μ < 0 && data.μ < minimum(data.λ)
            # The etrs_boundary algorithm is in a "stationary" point
            # with the most negative langrange multiplier being the
            # one for the constraint ‖x‖ = r
            # Thus we switch to the GeneralQP solver
            update_data!(interior_data, boundary_data)
            data = interior_data
        end
    end

    if isa(data, Data)
        return data.x, [data.λ; data.μ]
    else
        return data.x, [data.λ; zero(T)]
    end
end

function create_data(P, q, A, b, r, x_init; kwargs...)
    interior_data = GeneralQP.Data(P, q, A, b, x_init; kwargs..., r_max=r)
    # interior_data.verbosity = 0
    boundary_data = Data(P, q, A, b, r, x_init,
        interior_data.F.QR, interior_data.working_set, interior_data.ignored_set,
        interior_data.A_shuffled, interior_data.b_shuffled; kwargs...)
    return boundary_data, interior_data
end

function update_data!(qp::GeneralQP.Data, etrs::Data)
    qp.iteration = etrs.iteration
    # We only have to recompute qp.F.LDL
    # Adding a (temporarily) constraint parallel to x ensures that the
    # projected hessian is p.d.
    add_constraint!(etrs.F, etrs.x/norm(etrs.x))
    qp.x = etrs.x
    qp.F.m = qp.F.QR.n - qp.F.QR.m
    qp.F.artificial_constraints = 0  # No need for them now :)
    GeneralQP.update_views!(qp.F)
    qp.F.U.data .= cholesky!(Symmetric(etrs.F.ZPZ[qp.F.m:-1:1, qp.F.m:-1:1])).U
    qp.F.d .= 1
    # remove the temporal constraint parallel to x
    remove_constraint!(qp.F, qp.F.QR.m)
    GeneralQP.update_views!(qp)
    if qp.verbosity > 0
        @info "Changing to QP solver."
        print_header(qp)
        # (mod(qp.iteration + 1, qp.printing_interval) != 0) && print_info(qp)
        print_info(qp)
    end
end

function update_data!(etrs::Data, qp::GeneralQP.Data)
    if etrs.verbosity > 0
        @info "Changing to constant-norm QP solver."
    end
    etrs.iteration = qp.iteration
    etrs.x = qp.x
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
            # Check if idx corresponds to ||x|| = r?
            !etrs.done && idx <= length(etrs.working_set) && remove_constraint!(etrs, idx)
        end
    end
    GeneralQP.update_views!(etrs)
    if etrs.verbosity > 0
        print_header(etrs)
        # (mod(etrs.iteration + 1, etrs.printing_interval) != 0) && print_info(etrs)
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