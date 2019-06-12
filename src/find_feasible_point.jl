function find_feasible_point(A::Matrix{T}, b::Vector{T}, r_min::T=zero(T), r_max::T=T(Inf)) where T
	# Attempts to solve the feasibility problem
	# find       x
	# such that  Ax ≤ b
	#            r_min ≤ ‖x‖ ≤ r_max

	# First use CPLEX to find the solution x_min of
	# minimize     x'x
	# subject to   Ax ≤ b
	m, n = size(A)
	model = Model(with_optimizer(CPLEX.Optimizer))
    @variable(model, x[1:n])
    @objective(model, Min, dot(x, x))
	@constraint(model, A*x .<= b)
	optimize!(model)
    x_min = value.(x)
	# Check that x_min is feasible and ‖x_min‖ ≤ r_max
	@assert norm(x_min) <= r_max string("The problem is infeasible. Min radius:", norm(x_min))
	@assert maximum(A*x_min - b) <= 1e-11 string("Min-radius solution returned by Gurobi violates constraints by:", maximum(A*x_min - b))

	# Use GeneralQP to find a solution x
	# maximize     x'x
	# subject to   Ax ≤ b
	# that has radius less than or equal to (r_min + r_max)/2
	x = GeneralQP.solve(Matrix(-one(T)*I, n, n), zeros(T, n), A, b, x_min; r_max=(r_min+r_max)/2, printing_interval=5000)
	# Check that r_min ≤ ‖x‖ ≤ r_max and Ax ≤ b
	@assert norm(x) - r_min >= -1e-9 "Failed to find a feasible point"
	@assert norm(x) - r_max <= 1e-9 "Failed to find a feasible point"
	@assert maximum(A*x - b) <= 1e-9 "Failed to find a feasible point"

	return x
end