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
path = "/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/MarosMeszaros.jl/data/"
cd(path)
files = glob("*.jld2")
cd(working_dir)
success = 0
fail = 0
f_ipopt = zeros(0)
f_active = zeros(0)
for file in files
    # file = "DPKLO1.jld2"
    # file = "HS35MOD.jld2"  # LICQ failing
    # file = "GOULDQP2.jld2"
    # file = "PRIMALC2.jld2"
    # file = "QFORPLAN.jld2" Huge r
    # file = "QSCFXM2.jld2"
    # file = "QISRAEL.jld2" # Never finishes
    # file = "QPCSTAIR.jld2" # Never finishes
    # file = "QSTAIR.jld2"
    # file = "MOSARQP2.jld2" # Circle line can't find solutions
    # file = "QPCBLEND.jld2"
    # file = "HS35MOD.jld2" # LICQ fails - very small problem
    # file = "MOSARQP2.jld2" # LICQ almost fails - bigger problem
    # file = "QE226.jld2" # No TRS optimizer found when tol_hard=2e-7
    # file = "QETAMACR.jld2" # trs_boundary_small reporting no solutions?
    # file = "QBANDM.jld2" # new_contraint == NaN after binary search
    # file = "QBORE3D.jld2" # trs_boundary_small fails (singular matrix in extract_solution_hard_case)
    file = "GOULDQP2.jld2"
    filepath = string(path, file)
    if file != "nonzeros.jld2"#  && file != "QISRAEL.jld2" && file != "QPCSTAIR.jld2" && file != "QE226.jld2" && file != "GOULDQP2.jld2"
        q = load(filepath, "q")
        if length(q) < 1000
            P, q, A, b = load(filepath, "P", "q", "A", "b")
            m, n = size(A)
            if n < 1501
                #=
                sense=:Min
                solver = sense == :Min ? CplexSolver(CPX_PARAM_LPMETHOD=4, CPX_PARAM_BARCROSSALG=-1, CPX_PARAM_REDUCE=0, CPX_PARAM_RELAXPREIND=0, CPXPARAM_Preprocessing_Relax=0, CPXPARAM_Preprocessing_Reduce=0) : CplexSolver(CPXPARAM_OptimalityTarget=2)
                model = MathProgBase.LinearQuadraticModel(solver)
                MathProgBase.loadproblem!(model, A, -Inf*ones(n), Inf*ones(n), zeros(n), -Inf*ones(m), b, sense)
                MathProgBase.optimize!(model)
                x_init = MathProgBase.getsolution(model)
                =#
                model = JuMP.Model()
                setsolver(model, GurobiSolver(OutputFlag=0, FeasibilityTol=1e-9)) #, print_level=0))
                @variable(model, x[1:n])
                @constraint(model, A*x .<= b)
                status = JuMP.solve(model)
                x_init = getvalue(x)
                r = norm(x_init)
                if r > 1e-6 && r < 1e5 && status == :Optimal
                    @show file r
                    #=
                    x_ipopt = solve_ipopt(P, q, A, b, r)
                    @show 0.5*dot(x_ipopt, P*x_ipopt) + dot(q, x_ipopt)
                    =#
                    P = Matrix(P)
                    A = Matrix(A)
                    return eTRS.solve_boundary(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=1);
                    try
                        x_ipopt = solve_ipopt(P, q, A, b, r)
                        P = Matrix(P)
                        A = Matrix(A)
                        x = eTRS.solve_boundary(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=500, max_iter=5000)
                        append!(f_active, 0.5*dot(x, P*x) + dot(q, x))
                        append!(f_ipopt, 0.5*dot(x_ipopt, P*x_ipopt) + dot(q, x_ipopt))
                        global success += 1
                        #=
                        eTRS.solve_boundary(P, q, A, b, r, copy(x_init), verbosity=0, printing_interval=100, max_iter=5000);
                        @show "SUCCESS"
                        =#
                    catch e
                        # return eTRS.solve_boundary(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=1);
                        print(e)
                        @show "FAIL"
                        global fail += 1
                    end
                end
            end
        end
    end
end
diff = (f_active - f_ipopt)./abs.(f_active)
important_diff = diff[abs.(diff) .> 0.01]
println("IPOPT was better on: ", sum(important_diff .> 0), " problems.")
println("Active set was better on: ", sum(important_diff .< 0), " problems.")
println("There were: ", length(diff) - length(important_diff), " ties.")