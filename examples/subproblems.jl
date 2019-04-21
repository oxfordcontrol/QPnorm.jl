using LinearAlgebra
using JuMP, Gurobi, GeneralQP

function find_feasible_point(A::Matrix{T}, b::Vector{T}, r::T) where T
	# Attempts to solve the feasibility problem
	# find       x
	# such that  Ax ≤ b
	#            ‖x‖ = r

	# First use Gurobi to find the solution x_min of
	# minimize     x'x
	# subject to   Ax ≤ b
	m, n = size(A)
	model = Model(with_optimizer(Gurobi.Optimizer))
    @variable(model, x[1:n])
    @objective(model, Min, dot(x, x))
	@constraint(model, A*x .<= b)
	optimize!(model)
    x_min = value.(x)
	# Check that x_min is feasible and ‖x‖ ≤ r
	@assert norm(x_min) <= r string("The problem is infeasible. Min radius:", norm(x_min))
	@assert maximum(A*x_min - b) <= 0 string("Min-radius solution returned by Gurobi violates constraints by:", maximum(A*x_min - b))

	# Use GeneralQP to find a solution x
	# maximize     x'x
	# subject to   Ax ≤ b
	# that has radius less than or equal to r
	x = GeneralQP.solve(Matrix(-one(T)*I, n, n), zeros(T, n), A, b, x_min; r_max=r, printing_interval=5000)
	# Check that ‖x‖ = r and Ax ≤ b
	@assert abs(norm(x) - r) <= 1e-9 && maximum(A*x - b) <= 1e-9 "Failed to find a feasible point"

	return x
end