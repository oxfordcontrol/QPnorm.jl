using JuMP, Ipopt, Polynomials
using MathProgBase, CPLEX

function find_feasible_point(A, b, r, tol=1e-12)
	# Attempts to solve the feasibility problem
	# Ax <= b
	# ||x|| = r
	m, n = size(A)
	x_min = optimize_radius(A, b, r; sense=:Min)
	@assert norm(x_min) <= r string("The problem is infeasible. Min radius:", norm(x_min))
	@assert maximum(A*x_min - b) <= tol string("Min-radius solution violates constraints by:", maximum(A*x_min - b))
	x_max = optimize_radius(A, b, r; sense=:Max)
	@assert norm(x_max) >= r string("Could not find a feasible point. Max radius:", norm(x_max))
	@assert maximum(A*x_max - b) <= tol string("Max-radius solution violates constraints by:", maximum(A*x_max - b))

	d = x_max - x_min
	alpha = roots(Poly([norm(x_min)^2 - r^2, 2*d'*x_min, norm(d)^2]))
	alpha = alpha[alpha .>= 0][1]
	x = x_min + alpha*d
	@assert abs(norm(x) - r) <= tol
	@assert maximum(A*x - b) <= tol
	return x
end

function generate_feasible_point(A, b, tol=1e-12)
	m, n = size(A)
	x_min = optimize_radius(A, b; sense=:Min)
	@assert maximum(A*x_min - b) <= tol string("Min-radius solution violates constraints by:", maximum(A*x_min - b))
	x_max = similar(x_min); r = norm(x_max)
	try
		@show "here"
		x_max = optimize_radius(A, b; sense=:Max)
		r = (norm(x_min) + norm(x_max))/2
	catch
		# Unbounded max-radius
		r = 2*norm(x_min)
		@show "here"
		x_max = optimize_radius(A, b, r; sense=:Max)
	end

	@assert norm(x_max) >= r string("Could not find a feasible point. Max radius:", norm(x_max))
	@assert maximum(A*x_max - b) <= tol string("Max-radius solution violates constraints by:", maximum(A*x_max - b))

	d = x_max - x_min
	alpha = roots(Poly([norm(x_min)^2 - r^2, 2*d'*x_min, norm(d)^2]))
	alpha = alpha[alpha .>= 0][1]
	x = x_min + alpha*d
	@assert abs(norm(x) - r) <= tol
	@assert maximum(A*x - b) <= tol
	return x
end

function optimize_radius(A, b, r_max=Inf; sense=:Min)
	m, n = size(A)
	solver = sense == :Min ? CplexSolver(CPX_PARAM_QPMETHOD=1) : CplexSolver(CPXPARAM_OptimalityTarget=2)

	model = MathProgBase.LinearQuadraticModel(solver)
	MathProgBase.loadproblem!(model, A, -r_max*ones(n), r_max*ones(n), zeros(n), -Inf*ones(m), b, sense)
	MathProgBase.setquadobj!(model, SparseMatrixCSC(1.0*I, n, n))
	MathProgBase.optimize!(model)

	return MathProgBase.getsolution(model)
end

function solve_ipopt(P, q, A, b, r, x_init)
	n = size(P, 1)
	model = JuMP.Model()
	setsolver(model, IpoptSolver(max_iter=2000))#, print_level=0))
	@variable(model, x[1:n])
	for i in 1:n
		setvalue(x[i], x_init[i]);
	end
    @objective(model, Min, 0.5*dot(x, P*x) + dot(q,x))
    @constraint(model, A*x .<= b)
    @constraint(model, dot(x, x) <= r^2)
	status = JuMP.solve(model)
	# @assert status == :Optimal status
	objective = JuMP.getobjectivevalue(model)
	return getvalue(x)
end