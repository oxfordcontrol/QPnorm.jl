using QPnorm
using Random

rng = MersenneTwister(123)
n = 30
m = 50
P = randn(rng, n, n); P = (P + P')/2
q = randn(rng, n)
A = randn(rng, m, n); b = randn(rng, m)
r_min = 10.0
r_max = 50.0

x = find_feasible_point(A, b, r_min, r_max)
@elapsed x1 = solve(P, q, A, b, x, r_min=r_min, r_max=r_max, verbosity=1, printing_interval=1)