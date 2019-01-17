using LinearAlgebra, Random
using BenchmarkTools
using JLD2, FileIO
using JuMP
using Gurobi
using CPLEX
using Glob
# include("/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/GeneralQP.jl/src/GeneralQP.jl")
# using Main.GeneralQP
include("../src/eTRS.jl")
include("./subproblems.jl")
using MathProgBase
using Main.eTRS

working_dir = pwd()
path = "/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/CUTEst.jl/data/MASTSIF/"
cd(path)
files = glob("*.jld2")
cd(working_dir)
success = 0
fail = 0
f_ipopt = zeros(0)
f_active = zeros(0)
# for file in files
    # file = "FERRISDC.jld2"
    file = "EXPFITA.jld2"
    filepath = string(path, file)
    q = load(filepath, "q")
    P, q, A, b = load(filepath, "P", "q", "A", "b")
    m, n = size(A)
    @show m, n

    model = JuMP.Model()
    setsolver(model, GurobiSolver(OutputFlag=0, FeasibilityTol=1e-9)) #, print_level=0))
    @variable(model, x[1:n])
    @constraint(model, A*x .<= b)
    status = JuMP.solve(model)
    x_init = getvalue(x)

    model = JuMP.Model()
    setsolver(model, GurobiSolver(OutputFlag=0, FeasibilityTol=1e-9)) #, print_level=0))
    @variable(model, x[1:n])
    @objective(model, Min, dot(x, x))
    @constraint(model, A*x .<= b)
    status = JuMP.solve(model)
    x_init = getvalue(x)
    #=
    @show sort((A*x_init - b)[:])
    @show x_init
    @show m, n
    @show sum((A*x_init - b) .== 0.0)
    =#

    r = 10.0
    if norm(x_init) <= r
        # @assert norm(x_init) <= r norm(x_init) r
        @show file norm(x_init)
        # x = solve_ipopt(P, q, A, b, r, x_init)
        # @time x = solve_ipopt(P, q, A, b, r, x_init)
        P = Matrix(P)
        A = Matrix(A)
        # @show file r
        eTRS.solve(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=1000, max_iter=5000)
        #=
        try
            eTRS.solve(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=1000, max_iter=5000)
        catch e
            print(e)
        end
        =#
    end
    # @time eTRS.solve(P, q, A, b, r, x_init, verbosity=0, printing_interval=1)
# end
diff = (f_active - f_ipopt)./abs.(f_active)
important_diff = diff[abs.(diff) .> 0.01]
println("IPOPT was better on: ", sum(important_diff .> 0), " problems.")
println("Active set was better on: ", sum(important_diff .< 0), " problems.")
println("There were: ", length(diff) - length(important_diff), " ties.")