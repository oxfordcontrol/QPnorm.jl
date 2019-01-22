using JuMP, Ipopt, Polynomials

function find_feasible_point(A, b, r)
	# Attempts to solve the feasibility problem
	# Ax <= b
	# ||x|| = r
	m, n = size(A)
	x_min = optimize_radius(A, b, r; minimize=true)
	@assert norm(x_min) <= r string("The problem is infeasible. Min radius:", norm(x_min))
	@assert maximum(A*x_min - b) <= 0 string("Min-radius solution returned by IPOPT violates constraints by:", maximum(A*x_min - b))
	x_max = optimize_radius(A, b, r; minimize=false)
	@assert norm(x_max) >= r string("Could not find a feasible point. Max radius:", norm(x_max))
	@assert maximum(A*x_max - b) <= 0 string("Max-radius solution returned by IPOPT violates constraints by:", maximum(A*x_max - b))

	d = x_max - x_min
	alpha = roots(Poly([norm(x_min)^2 - r^2, 2*d'*x_min, norm(d)^2]))
	alpha = alpha[alpha .>= 0][1]
	x = x_min + alpha*d
	@assert abs(norm(x) - r) <= 1e-9*r
	@assert maximum(A*x - b) <= 1e-7
	return x
end

function optimize_radius(A, b, r; minimize=true)
	n = size(A, 2)
	model = JuMP.Model()
    setsolver(model, IpoptSolver(max_iter=500, bound_relax_factor=0.0, print_level=0))
	@variable(model, x[1:n])
	objective = dot(x, x)
	if !minimize
		objective = -objective
	end
	@objective(model, Min, objective)
    @constraint(model, A*x - b .<= 0)
    @constraint(model, x .<= 5*r)
    @constraint(model, x .>= -5*r)
	status = JuMP.solve(model)
	return getvalue(x)
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