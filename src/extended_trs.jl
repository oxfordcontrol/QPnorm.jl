function solve(P::AbstractMatrix{T}, q::Vector{T}, A::AbstractMatrix{T}, b::Vector{T}, r::T,
    x::Vector{T}; max_iter=Inf, kwargs...) where T 

    @show @elapsed boundary_data, interior_data = create_data(P, q, A, b, r, x; kwargs...)

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
    # @show boundary_data.time_trs boundary_data.time_gradient boundary_data.time_move boundary_data.time_add boundary_data.time_remove boundary_data.time_kkt
    return data.x
end

function create_data(P, q, A, b, r, x_init; kwargs...)
    interior_data = GeneralQP.Data(Matrix(P), q, Matrix(A), b, x_init; kwargs..., r_max=r)
    interior_data.verbosity = 0
    boundary_data = Data(P, q, A, b, r, x_init,
        interior_data.working_set, interior_data.ignored_set; kwargs...)
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
    if etrs.n - length(etrs.working_set) <= 1
        fact(etrs)
        etrs.x0 = etrs.x - project(data, data.x)
        if sqrt(etrs.r^2 - norm(etrs.x0)^2) <= 1e-10
            @warn "LICQ failed. Terminating"
            etrs.done = true
        else
            idx = check_kkt!(etrs)
            # Check if idx corresponds to ||x|| = r?
            !etrs.done && idx <= length(etrs.working_set) && remove_constraint!(etrs, idx)
        end
    end
    # GeneralQP.update_views!(etrs)
    fact(etrs)
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