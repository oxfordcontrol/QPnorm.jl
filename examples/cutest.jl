using LinearAlgebra, Random
using BenchmarkTools
using JLD2, FileIO
using JuMP
using Glob
using SparseArrays
include("../src/eTRS.jl")
include("./subproblems.jl")
include("./solve_ipopt.jl")
using Main.eTRS
using DataFrames
using CSV

working_dir = pwd()
path = "./data/"
cd(path)
files = glob("*.jld2")
cd(working_dir)

function compute_metrics(P, q, A, b, r, x, λ) #, inactive_set)
    m, n = size(A)
    grad_residual = norm(P*x + q + A'*λ[1:end-1] + λ[end]*x, Inf)
    active_set = λ .>= 1e-8
    A_active = [A; x'][active_set, :]
    if length(A_active) > 0
        V = nullspace([A; x'][active_set, :])
    else
        V = diagm(0 => ones(n))
    end
    # Q = qr([A' x][:, inactive_set]).Q*Matrix(I, n, n)
    # V = Q[:, sum(inactive_set)+1:end]
    if length(V) > 0
        min_eig = minimum(eigvals(Symmetric(V'*(P + I*λ[end])*V)))
    else
        min_eig = 0.0
    end
    f = dot(x, P*x)/2 + dot(x, q)
    infeasibility = max(maximum(A*x - b), norm(x) - r, 0)
    complementarity = norm(λ.*[A*x - b; norm(x) - r], Inf)
    dual_infeasibility = max(-minimum(λ), 0.0)

    return f, grad_residual, infeasibility, dual_infeasibility, complementarity, min_eig
end

df = DataFrame(name=String[], n = Int[], m = Int[],
    time = Float64[], time_ipopt = Float64[],
    objective = Float64[], objective_ipopt = Float64[],
    infeasibility = Float64[], infeasibility_ipopt = Float64[],
    dual_infeasibility = Float64[], dual_infeasibility_ipopt = Float64[],
    complementarity = Float64[], complementarity_ipopt = Float64[],
    grad_residual = Float64[], grad_residual_ipopt = Float64[],
    min_eig = Float64[], min_eig_ipopt = Float64[]
)
#=
    multipliers = [], multipliers_ipopt = [],
    solution = [], solution_ipopt = []
    )
=#

i = 0
for file in files[1:end]
    global i += 1
    filepath = string(path, file)
    q = load(filepath, "q")
    P, q, A, b, x0 = load(filepath, "P", "q", "A", "b", "x0")
    # We want to solve
    # minimize    ½x'Px + q'x + f0
    # subject to  Ax ≤ b
    #             ‖x - x0‖ = r
    # We convert the problem to a standard form by a change of variables z = x - x0
    q = q + P*x0
    b = b - A*x0
    # f0 += dot(data.x0, (P*data.x0))/2 + dot(q, x0)
    x_init = zeros(size(x0))
    m, n = size(A)
    @show m, n
    println("Solving problem #", i, " : ", file[1:end-5], " with norm(x_init)=", norm(x_init))
    r = 1.0
    x_ipopt = Float64[]; f_ipopt = NaN; infeasibility_ipopt = NaN; t_ipopt = NaN
    grad_residual_ipopt = NaN; min_eig_ipopt = NaN; multipliers_ipopt = Float64[];
    complementarity_ipopt = NaN; dual_infeasibility_ipopt = NaN
    try
        x_ipopt, multipliers_ipopt, t_ipopt = solve_ipopt(P, q, A, b, r, 0, copy(x_init); print_level=0)
        x_ipopt, multipliers_ipopt, t_ipopt = solve_ipopt(P, q, A, b, r, 0, copy(x_init); print_level=0)

        f_ipopt, grad_residual_ipopt, infeasibility_ipopt, dual_infeasibility_ipopt, complementarity_ipopt, min_eig_ipopt = compute_metrics(Matrix(P), q, Matrix(A), b, r, x_ipopt, multipliers_ipopt)
    catch e
        nothing
    end

    x = Float64[]; f = NaN; infeasibility = NaN; t = NaN
    grad_residual = NaN; min_eig = NaN; multipliers = Float64[];
    complementarity = NaN; dual_infeasibility = NaN
    P = Matrix(P); A = Matrix(A)
    try
        t = @elapsed x, multipliers = eTRS.solve(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=5000, max_iter=5000)
        t = @elapsed x, multipliers = eTRS.solve(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=5000, max_iter=5000)
        # inactive_set = multipliers .!== 0.0
        f, grad_residual, infeasibility, dual_infeasibility, complementarity, min_eig = compute_metrics(P, q, A, b, r, x, multipliers)
    catch e
        nothing
    end

    push!(df, [file[1:end-5], n, m,
            t, t_ipopt,
            f, f_ipopt,
            infeasibility, infeasibility_ipopt,
            dual_infeasibility, dual_infeasibility_ipopt,
            complementarity, complementarity_ipopt,
            grad_residual, grad_residual_ipopt,
            min_eig, min_eig_ipopt])
            #=
            multipliers, multipliers_ipopt,
            x, x_ipopt])
            =#
    
    df |> CSV.write(string("statistics.csv"))
end
