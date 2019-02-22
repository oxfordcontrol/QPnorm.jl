using Printf

function print_header(data::Data{T}) where T
    @printf("Iter \t  Objective  \t Inf Linear \t Inf Sphere \t Gradient res \t L/HS \t TRS \t AC\n")
end

function print_info(data::Data{T}) where T
    n = Int(length(data.x)/2)
    x_ = data.x[1:n] - data.x[n+1:end]
    @printf("%d \t  %.5e \t %.5e \t %.5e \t %s \t %d/%d \t %c \t %d\n",
        data.iteration,
        dot(x_, data.S*x_)/2,
        max(-minimum(data.x), sum(data.x) - data.gamma, maximum(abs.(data.W'*data.x))),
        abs(1.0 - norm(data.x)),
        isnan(data.residual) ? string(@sprintf("%.5e", data.residual), "   ") : @sprintf("%.5e", data.residual),
        data.grad_steps, data.eigen_steps,
        data.trs_choice,
        length(data.zero_indices) + data.gamma_active
    )
    data.residual = NaN
    data.grad_steps = data.eigen_steps = 0;
end