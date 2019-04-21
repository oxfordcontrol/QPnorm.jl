include("./subproblems.jl")
include("./solve_ipopt.jl")
using eTRS
using Random
using DataFrames, CSV
using JLD2, FileIO
using Glob

rng = MersenneTwister(123)

function compute_metrics(P::Matrix{T}, q::Vector{T},
    A::Matrix{T}, b::Vector{T}, r::T,
    x::Vector{T}, λ::Vector{T}) where {T}

    m, n = size(A)

    f = dot(x, P*x)/2 + dot(x, q)

    grad_residual = norm(P*x + q + A'*λ[1:end-1] + λ[end]*x, Inf)
    infeasibility = max(maximum(A*x - b), abs(norm(x)^2 - r^2), 0)
    complementarity = maximum(minimum([abs.(λ[1:end-1]) abs.(A*x - b)], dims=2))
    dual_infeasibility = max(-minimum(λ[1:end-1]), 0.0)

    active_set = λ .>= 1e-8
    active_set[end] = true # Norm constraint is an equality constraint; thus it should always be included
    V = nullspace([A; x'][active_set, :])
    if length(V) > 0
        min_eig = minimum(eigvals(Symmetric(V'*(P + I*λ[end])*V)))
    else
        min_eig = 0.0
    end

    return f, grad_residual, infeasibility, dual_infeasibility, complementarity, min_eig
end

df = DataFrame(n = Int[], m = Int[],
    time = Float64[], time_ipopt = Float64[],
    objective = Float64[], objective_ipopt = Float64[],
    infeasibility = Float64[], infeasibility_ipopt = Float64[],
    dual_infeasibility = Float64[], dual_infeasibility_ipopt = Float64[],
    complementarity = Float64[], complementarity_ipopt = Float64[],
    grad_residual = Float64[], grad_residual_ipopt = Float64[],
    min_eig = Float64[], min_eig_ipopt = Float64[]
)

working_dir = pwd()
path = "./random_data/"
cd(path); files = glob("*.jld2"); cd(working_dir)

for file in files
    P, q, A, b, r, x_init = load(string(path, file), "P", "q", "A", "b", "r", "x_init")
    m, n = size(A)

    x_ipopt = Float64[]; f_ipopt = NaN; infeasibility_ipopt = NaN; t_ipopt = NaN
    grad_residual_ipopt = NaN; min_eig_ipopt = NaN; multipliers_ipopt = Float64[];
    complementarity_ipopt = NaN; dual_infeasibility_ipopt = NaN
    try
        x_ipopt, multipliers_ipopt, t_ipopt = solve_ipopt(P, q, A, b, r, 0, copy(x_init); print_level=0)
        f_ipopt, grad_residual_ipopt, infeasibility_ipopt, dual_infeasibility_ipopt, complementarity_ipopt, min_eig_ipopt = compute_metrics(Matrix(P), q, Matrix(A), b, r, x_ipopt, multipliers_ipopt)
    catch e
        nothing
    end

    x = Float64[]; f = NaN; infeasibility = NaN; t = NaN
    grad_residual = NaN; min_eig = NaN; multipliers = Float64[];
    complementarity = NaN; dual_infeasibility = NaN
    try
        t = @elapsed x, multipliers = eTRS.solve_boundary(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=5000, max_iter=5000)
        t = @elapsed x, multipliers = eTRS.solve_boundary(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=5000, max_iter=5000)
        f, grad_residual, infeasibility, dual_infeasibility, complementarity, min_eig = compute_metrics(P, q, A, b, r, x, multipliers)
    catch e
        nothing
    end

    push!(df, [n, m,
            t, t_ipopt,
            f, f_ipopt,
            infeasibility, infeasibility_ipopt,
            dual_infeasibility, dual_infeasibility_ipopt,
            complementarity, complementarity_ipopt,
            grad_residual, grad_residual_ipopt,
            min_eig, min_eig_ipopt])

    df |> CSV.write(string("results_random.csv"))
end