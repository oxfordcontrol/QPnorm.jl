include("../src/eTRS.jl")
include("./subproblems.jl")

using LinearAlgebra, Random
using BenchmarkTools
using JLD2
using Main.eTRS

# @load string("/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/MarosMeszaros.jl/data/Q25FV47.jld2") P q A b f0 
@load string("/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/MarosMeszaros.jl/data/EXDATA.jld2") P q A b f0 

@show size(P)

#=
model = Model(solver=GurobiSolver(Method=3))
@variable(model, x[1:size(A, 2)])
@constraint(model, A*x - b .<=0)
@objective(model, Min, x'*x)
status = JuMP.solve(model)
x_init = getvalue(x)  # Initial point to be passed to our solver
=#

r = 200.0
P = P# - 50*I
# solve_ipopt(P, q, A, b, r)
x_init = find_feasible_point(A, b, r)
@save "x_init.jld2" x_init r
# @load "x_init.jld2" x_init r

# @show maximum(A*x_init - b)
# @show sum(A*x_init - b .>= -1e-11)
residuals = A*x_init - b
# using PyPlot
# plotly() # Choose the Plotly.jl backend for web interactivity
# semilogy(-residuals[sortperm(-residuals)])

eTRS.solve_boundary(Matrix(P), q, Matrix(A), b, r, copy(x_init), printing_interval=100);
# @time active_set_solver(Matrix(P), q, Matrix(A), b, r, copy(x_init))
