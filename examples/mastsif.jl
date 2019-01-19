using LinearAlgebra, Random
using BenchmarkTools
using JLD2, FileIO
using JuMP
using Gurobi
using CPLEX
using Glob
include("../src/eTRS.jl")
include("./subproblems.jl")
using Main.eTRS

working_dir = pwd()
path = "/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/CUTEst.jl/data/MASTSIF/"
cd(path)
files = glob("*.jld2")
cd(working_dir)

for file in files
    # EXPFITA FERRISDC BLOCKQP3
    # file = string("BLOCKQP5", ".jld2")
    filepath = string(path, file)
    q = load(filepath, "q")
    P, q, A, b = load(filepath, "P", "q", "A", "b")
    m, n = size(A)
    @show m, n
    
    # Get Initial feasible point
    model = JuMP.Model()
    setsolver(model, GurobiSolver(OutputFlag=0, FeasibilityTol=1e-9)) #, print_level=0))
    @variable(model, x[1:n])
    @objective(model, Min, dot(x, x))
    @constraint(model, A*x .<= b)
    status = JuMP.solve(model)
    x_init = getvalue(x)

    println("Solving problem: ", file[1:end-5], " with norm(x_init)=", norm(x_init))
    r = 10.0
    if norm(x_init) <= r
        # x = solve_ipopt(P, q, A, b, r, x_init)
        # @show file r
        P = Matrix(P); A = Matrix(A)
        eTRS.solve(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=500, max_iter=5000)
    end
end