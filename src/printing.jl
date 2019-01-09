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

function debug_gradient_step(a1, a2, b1, r, y_l, y_g, f0, f_2d)
    print_tol = 1e-5
    @printf("------------------------------\n")
    @printf("Gradient steps failed! \n")
    @printf("The problem might be badly scaled or it might belong to the hard case.\n")
    @printf("Debug info:\n")
    @printf("------------------------------\n")
    @printf("Id \t Delta \t\t Infeasibility\n")
    @printf("%c \t %.5e \t %.5e\n", 'g', f_2d(y_g[1], y_g[2]) - f0, maximum(a1*y_g[1] + a2*y_g[2] - b1))
    @printf("%c \t %.5e \t %.5e\n", 'l', f_2d(y_l[1], y_l[2]) - f0, maximum(a1*y_l[1] + a2*y_l[2] - b1))
    min_value = (1 + sign(f0)*1e-7)*f0
    for i = 1:length(a1)
        y11, y12, y21, y22 = circle_line_intersections(a1[i], a2[i], b1[i], r)
        if maximum(a1*y11 + a2*y12 - b1) <= print_tol
            @printf("%d \t %.5e \t %.5e\n", i, f_2d(y11, y12) - f0, maximum(a1*y11 + a2*y12 - b1))
        end
        if maximum(a1*y21 + a2*y22 - b1) <= print_tol
            @printf("%d \t %.5e \t %.5e\n", i, f_2d(y21, y22) - f0, maximum(a1*y21 + a2*y22 - b1))
        end
    end
    @printf("------------------------------\n")

    return false
end