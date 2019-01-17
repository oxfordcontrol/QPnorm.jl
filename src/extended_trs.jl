function solve(P::Matrix{T}, q::Vector{T}, A::Matrix{T}, b::Vector{T}, r::T,
    x::Vector{T}; kwargs...) where T 

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

    while !data.done && data.iteration <= Inf
        iterate!(data)
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
    return x
end

function create_data(P, q, A, b, r, x_init; kwargs...)
    interior_data = GeneralQP.Data(P, q, A, b, x_init; kwargs..., r_max=r)
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
    qp.F.U.data .= cholesky!(Symmetric(etrs.F.ZPZ)).U
    qp.F.d .= 1
    qp.F.artificial_constraints = 0  # No need for them now :)
    GeneralQP.update_views!(qp.F)
    # remove the temporal constraint parallel to x
    remove_constraint!(qp.F, qp.F.QR.m)
    GeneralQP.update_views!(qp)
    print_header(qp)
end

function update_data!(etrs::Data, qp::GeneralQP.Data)
    etrs.iteration = qp.iteration
    etrs.x = qp.x
    # We only have to recompute etrs.F.ZPZ
    etrs.F.m = etrs.F.QR.n - etrs.F.QR.m
    GeneralQP.update_views!(etrs.F)
    etrs.F.ZPZ .= etrs.F.Z'*etrs.F.P*etrs.F.Z
    etrs.F.ZPZ .= (etrs.F.ZPZ + etrs.F.ZPZ')/2
    if etrs.F.m <= 1
        idx = check_kkt!(etrs)
        # Check if idx corresponds to ||x|| = r?
        !etrs.done && idx <= length(etrs.working_set) && remove_constraint!(etrs, idx)
    end
    @show "change to etrs"
    GeneralQP.update_views!(etrs)
    print_header(etrs)
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
