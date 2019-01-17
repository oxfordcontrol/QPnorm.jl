include("../src/eTRS.jl")
include("./subproblems.jl")

using LinearAlgebra, Random
using BenchmarkTools
using JLD2
using Main.eTRS
using GeneralQP
using JuMP
using Gurobi

@load string("/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/MarosMeszaros.jl/data/Q25FV47.jld2") P q A b f0 
# @load string("/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/MarosMeszaros.jl/data/EXDATA.jld2") P q A b f0 

n = length(q)

@show size(P)

model = JuMP.Model()
setsolver(model, GurobiSolver()) #, print_level=0))
@variable(model, x[1:n])
@constraint(model, A*x .<= b)
status = JuMP.solve(model)
x_init = getvalue(x)
r = norm(x_init)
@save "x_init.jld2" x_init r

# @load "x_init.jld2" x_init r

#=
P = P# - 50*I
x_init = generate_feasible_point(A, b)
@time x_init = generate_feasible_point(A, b)
solve_ipopt(P, q, A, b, r)
@time solve_ipopt(P, q, A, b, r)
# @save "x_init.jld2" x_init r
@load "x_init.jld2" x_init r
=#

# @show maximum(A*x_init - b)
# @show sum(A*x_init - b .>= -1e-11)
residuals = A*x_init - b
# using PyPlot
# plotly() # Choose the Plotly.jl backend for web interactivity
# semilogy(-residuals[sortperm(-residuals)])

solve_ipopt(P, q, A, b, r)
@show r
P = Matrix(P)
A = Matrix(A)
eTRS.solve_boundary(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=200);
@time eTRS.solve_boundary(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=200);
# @time active_set_solver(Matrix(P), q, Matrix(A), b, r, copy(x_init))