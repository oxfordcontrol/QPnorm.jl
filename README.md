# eTRS.jl: Solving Trust Region Subproblems with Inequality Constraints

This is an active-set algorithm for solving problems of the form
```
minimize    ½x'Px + q'x
subject to  ‖x‖ ≤ r
            Ax ≤ b
```
where `x` in the `n-`dimensional variable and `P` is a general symmetric (definite/indefinite) matrix.

## Installation
The solver can be installed by running
```
add https://github.com/oxfordcontrol/GeneralQP.jl
```

## Usage
Problems of the form
```
minimize    ½x'Px + q'x
subject to  ‖x‖ ≤ r
            Ax ≤ b
```
can be solved with
```
solve(P, q, A, b, r, x_init; kwargs) -> x
```
with **inputs** (`T` is any real numerical type):

* `P::Matrix{T}`: the quadratic of the cost;
* `q::Vector{T}`: the linear cost;
* `A::Matrix{T}` and `b::AbstractVector{T}`: the constraints; and
* `r::T` the radius;
* `x_init::Vector{T}`: the initial, [feasible](#obtaining-an-initial-feasible-point) point

**keywords** (optional):
* `verbosity::Int=1` the verbosity of the solver ranging from `0` (no output)
to `2` (most verbose). Note that setting `verbosity=2` affects the algorithm's performance.
* `printing_interval::Int=50`.

and **output** `x::Vector{T}`, the calculated optimizer.

Constant norm problems
```
minimize    ½x'Px + q'x
subject to  ‖x‖ = r
            Ax ≤ b
```
can be solved with
```
solve_boundary(P, q, A, b, r, x_init; kwargs) -> x
```
with inputs identical to `solve(···)`.

## Obtaining an initial feasible point

An initial feasible point for the `‖x‖ ≤ r` can be obtained by solving the strongly-convex qp
```
minimize    x'x
subject to  Ax ≤ b
```
e.g. with a standard active-set method
```
using JuMP, Gurobi
# Choose Gurobi's primal simplex method
model = Model(solver=GurobiSolver(Presolve=0, Method=0))
@variable(model, x[1:size(A, 2)])
@constraint(model, A*x - b .<=0)
@objective(model, dot(x, x), objective)
status = JuMP.solve(model)

x_min_radius = getvalue(x)  # Initial point to be passed to our solver
```

Finding an initial feasible point for the `‖x‖ = r` is NP-complete in general. However, one can attempt to obtain a feasible point as following:
```
ToDo: code from subproblems.jl
```