include("./subproblems.jl")
using Random
using JLD2, FileIO

rng = MersenneTwister(123)

function create_problems()
    counter = 1
    dimensions = Vector(range(100, 2000, length=10))
    prepend!(dimensions, 10) # Small problem to warm up solvers
    r = 50.0
    for dimension in dimensions
        n = Int(floor(dimension))
        m = Int(floor(1.5*n))
        P = randn(rng, n, n); P = (P + P')/2
        q = randn(rng, n)
        A = randn(rng, m, n); b = randn(rng, m)
        x_init = find_feasible_point(A, b, r)

        @save string("random_data/", counter, ".jld2") P q A b r x_init
        counter += 1
    end
end

create_problems()