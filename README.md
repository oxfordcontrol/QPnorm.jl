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
```julia
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

and **output** `x::Vector{T}`, the calculated optimizer, and `λ::Vector{T}` that contains the Lagrange multipliers of the linear inequalities and the norm constraint.

## Obtaining an initial feasible point

Finding an initial feasible point for the problem considered in the repository is in general intractable. However the following function from `examples/subproblems.jl`
```julia
find_feasible_point(A, b, r_min=0, r_max=Inf) -> x
```
attempts to find a feasible point by first minimizing `x'x` and then maximizing `x'x ` over the polyhedron `Ax ≤ b`.