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
using Profile, ProfileView

working_dir = pwd()
path = "/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/CUTEst.jl/data/Maros-Meszaros/"
cd(path)
files = glob("*.jld2")
cd(working_dir)
success = 0
fail = 0
f_ipopt = zeros(0)
f_active = zeros(0)
k = 0
for file in files[1:end]
    # file = "QBRANDY.jld2"
    filepath = string(path, file)
    if file != "nonzeros.jld2"  && file != "GOULDQP2.jld2" # && file != "QPCSTAIR.jld2" && file != "QE226.jld2" && file != "GOULDQP2.jld2"
        q = load(filepath, "q")
        @show length(q)
        if length(q) > 5000 && length(q) < 30000
            P, q, A, b = load(filepath, "P", "q", "A", "b")
            m, n = size(A)
            model = JuMP.Model()
            setsolver(model, GurobiSolver(OutputFlag=0, FeasibilityTol=1e-9)) #, print_level=0))
            @variable(model, x[1:n])
            @constraint(model, A*x .<= b)
            status = JuMP.solve(model)
            x_init = getvalue(x)
            r = norm(x_init)
            @show m, n, r
            if r > 1e-6 && r < 1e5 && status == :Optimal
                global k += 1
                if k > 5
                @show file, r, m, n
                x_ipopt = solve_ipopt(P, q, A, b, r, x_init)
                eTRS.solve_boundary(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=100, max_iter=5000);
                #=
                Profile.clear()		
                Profile.@profile eTRS.solve_boundary(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=500, max_iter=5000);# return nothing
                Profile.clear()		
                Profile.@profile eTRS.solve_boundary(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=500, max_iter=5000);# return nothing
                ProfileView.view()		
                println("Press enter to continue...")		
                readline(stdin)		
                =#
                #=
                try
                    x_ipopt = solve_ipopt(P, q, A, b, r, x_init)
                    x = eTRS.solve_boundary(Matrix(P), q, Matrix(A), b, r, copy(x_init), verbosity=1, printing_interval=500, max_iter=5000)
                    append!(f_active, 0.5*dot(x, P*x) + dot(q, x))
                    append!(f_ipopt, 0.5*dot(x_ipopt, P*x_ipopt) + dot(q, x_ipopt))
                    global success += 1
                catch e
                    @show e
                    global fail += 1
                end
                =#
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