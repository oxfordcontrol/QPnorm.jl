# QPnorm.jl: Solving Quadratic Problems with a norm constraint

This is an active-set algorithm for solving problems of the form
```
minimize    ½x'Px + q'x
subject to  ‖x‖ ∈ [r_min, r_max]
            Ax ≤ b
```
where `x` in the `n-`dimensional variable and `P` is a symmetric (definite/indefinite) matrix and `‖.‖` is the 2-norm.

This repository is the official implementation of the [following paper](https://arxiv.org/abs/1906.04022):
```
Rontsis N., Goulart P.J., & Nakatsukasa, Y.
An active-set algorithm for norm constrained quadratic problems
Preprint in Arxiv
```

## Installation
The solver can be installed by running
```
add https://github.com/oxfordcontrol/QPnorm.jl
```
in [Julia's Pkg REPL mode](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html#Getting-Started-1).

This package is based on [TRS.jl](https://github.com/oxfordcontrol/TRS.jl) and [GeneralQP.jl](https://github.com/oxfordcontrol/GeneralQP.jl).

## Usage
Problems of the form
```
minimize    ½x'Px + q'x
subject to  ‖x‖ ∈ [r_min, r_max]
            Ax ≤ b
```
can be solved with
```
solve(P, q, A, b, r, x_init; kwargs) -> x
```
with **inputs** (`T` is any real numerical type):

* `P::Matrix{T}`: the quadratic cost;
* `q::Vector{T}`: the linear cost;
* `A::Matrix{T}` and `b::AbstractVector{T}`: the linear constraints;
* `r_min::T=zero(T)` and `r_max::T=T(Inf)`: the 2-norm bounds
* `x_init::Vector{T}`: the initial, [feasible](#obtaining-an-initial-feasible-point) point

**keywords** (optional):
* `verbosity::Int=1` the verbosity of the solver ranging from `0` (no output)
to `2` (most verbose). Note that setting `verbosity=2` affects the algorithm's performance.
* `max_iter=Inf`: Maximum number of iterations
* `printing_interval::Int=50`.

and **output** `x::Vector{T}`, the calculated optimizer.

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
