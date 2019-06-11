include("../src/QPnorm.jl")
include("../examples/subproblems.jl")
using Main.QPnorm
using Random, Test

function optimality_metrics(P, q, A, b, r_min, r_max, x, λ)
    m, n = size(A)

    f = dot(x, P*x)/2 + dot(x, q)

    grad_residual = norm(P*x + q + A'*λ[1:end-1] + λ[end]*x, Inf)
    infeasibility = max(maximum(A*x - b), norm(x)^2 - r_max^2, r_min^2 - norm(x)^2, 0)

    # Find active r, if any
    if abs(norm(x) - r_min) < abs(norm(x) - r_max)
        r = r_min
    else
        r = r_max
    end

    complementarity = maximum(minimum([abs.(λ) abs.([A*x - b; norm(x)^2 - r^2])], dims=2))
    dual_infeasibility = max(-minimum(λ), 0.0)

    active_set = λ .>= 1e-8
    A_active = [A; x'][active_set, :]
    if length(A_active) > 0
        V = nullspace([A; x'][active_set, :])
    else
        V = diagm(0 => ones(n))
    end
    if length(V) > 0
        min_eig = minimum(eigvals(Symmetric(V'*(P + I*λ[end])*V)))
    else
        min_eig = 0.0
    end

    return f, grad_residual, infeasibility, dual_infeasibility, complementarity, min_eig
end

rng = MersenneTwister(123)
tol = 1e-7
@testset "Optimality Conditions of a random problem" begin
    n = 100
    m = 200
    P = randn(rng, n, n); P = (P + P')/2;
    q = randn(rng, n)
    A = randn(rng, m, n)
    b = randn(rng, m)
    r_min = 10.0; r_max = 20.0;
    x_init = find_feasible_point(A, b, r_min, r_max)
    x, λ = Main.QPnorm.solve(P, q, A, b, x_init, r_min=r_min, r_max=r_max, printing_interval=100)
    f, grad, inf, dinf, compl, min_eig = optimality_metrics(P, q, A, b, r_min, r_max, x, λ)
    @test grad < tol
    @test inf < tol
    @test dinf < tol
    @test compl < tol
    @test -min_eig < tol
end