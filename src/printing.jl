using Printf

function print_header(data::Data{T}) where T
    @printf("Iter \t  Objective  \t Inf Linear \t Inf Sphere \t Gradient res \t L/HS \t TRS \t AC\n")
end

function print_info(data::Data{T}) where T
    @printf("%d \t  %.5e \t %.5e \t %.5e \t %s \t %d/%d \t %c \t %d\n",
        data.iteration,
        data.x'*data.P*data.x/2 + dot(data.q, data.x),
        max(maximum(data.A*data.x - data.b), 0),
        abs(data.r - norm(data.x)),
        isnan(data.residual) ? string(@sprintf("%.5e", data.residual), "   ") : @sprintf("%.5e", data.residual),
        data.grad_steps, data.eigen_steps,
        data.trs_choice,
        length(data.working_set)
    )
    data.residual = NaN
    data.grad_steps = data.eigen_steps = 0;
end