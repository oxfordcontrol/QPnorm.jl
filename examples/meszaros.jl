using LinearAlgebra, Random
using BenchmarkTools
using JLD2, FileIO
using JuMP
using Gurobi
using CPLEX
using Glob
include("../src/eTRS.jl")
include("./subproblems.jl")
using MathProgBase
using Main.eTRS

working_dir = pwd()
path = "/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/CUTEst.jl/data/Maros-Meszaros/"
cd(path)
files = glob("*.jld2")
cd(working_dir)
success = 0
fail = 0
f_ipopt = zeros(0)
f_active = zeros(0)
for file in files
    filepath = string(path, file)
    if file != "nonzeros.jld2"#  && file != "QISRAEL.jld2" && file != "QPCSTAIR.jld2" && file != "QE226.jld2" && file != "GOULDQP2.jld2"
        q = load(filepath, "q")
        if length(q) < 1501
            P, q, A, b = load(filepath, "P", "q", "A", "b")
            m, n = size(A)
            model = JuMP.Model()
            setsolver(model, GurobiSolver(OutputFlag=0, FeasibilityTol=1e-9)) #, print_level=0))
            @variable(model, x[1:n])
            @constraint(model, A*x .<= b)
            status = JuMP.solve(model)
            x_init = getvalue(x)
            r = norm(x_init)
            if r > 1e-6 && r < 1e5 && status == :Optimal
                @show file, r, m, n
                x = eTRS.solve_boundary(Matrix(P), q, Matrix(A), b, r, copy(x_init), verbosity=1, printing_interval=1, max_iter=5000)
                return nothing
                try
                    x_ipopt = solve_ipopt(P, q, A, b, r)
                    x = eTRS.solve_boundary(Matrix(P), q, Matrix(A), b, r, copy(x_init), verbosity=1, printing_interval=500, max_iter=5000)
                    append!(f_active, 0.5*dot(x, P*x) + dot(q, x))
                    append!(f_ipopt, 0.5*dot(x_ipopt, P*x_ipopt) + dot(q, x_ipopt))
                    global success += 1
                catch e
                    @show e
                    global fail += 1
                end
            end
        end
    end
end
diff = (f_active - f_ipopt)./abs.(f_active)
important_diff = diff[abs.(diff) .> 0.01]
@show success fail
println("IPOPT was better on: ", sum(important_diff .> 0), " problems.")
println("Active set was better on: ", sum(important_diff .< 0), " problems.")
println("There were: ", length(diff) - length(important_diff), " ties.")