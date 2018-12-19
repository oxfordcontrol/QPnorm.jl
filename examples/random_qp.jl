include("../src/eTRS.jl")
include("./subproblems.jl")
using Main.eTRS
using Random

rng = MersenneTwister(123)
n = 30
m = 50
P = randn(rng, n, n); P = (P + P')/2
q = randn(rng, n)
A = randn(rng, m, n); b = randn(rng, m)
r = 50.0

x = find_feasible_point(A, b, r)
# x = optimize_radius(A, b, r; minimize=true)
@elapsed x1 = Main.eTRS.solve(P, q, A, b, r, x)