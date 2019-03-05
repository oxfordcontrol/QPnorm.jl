using JuMP, Ipopt, Polynomials, LinearAlgebra, Statistics, Arpack
using SparseArrays
using LinearMaps
include("../src/eTRS.jl")
using Main.eTRS
using Random

function binary_search(S, Y, nz)
    @show @elapsed x_init = eTRS.get_initial_guess(S, Int(nz); Y=Y)
    high = norm(x_init, 1)
    low = high/2

    n = length(x_init)
    x_warm = x_init
    max_iter = 30
    for i = 1:max_iter
        gamma = (high - low)/2 + low
        y, data = eTRS.sparse_pca(S, gamma, x_warm; Y=Y, verbosity=2, printing_interval=3000, max_iter=10000);
        x_warm = data.x[1:n] - data.x[n+1:end]
        nonzeros = sum(abs.(y) .> 1e-7)
        println("Nonzeros: ", nonzeros, " γ: [", high, ", ", low, "]")
        if nonzeros == nz || i == max_iter
            println("Found at iteration:", i)
            return y, data
        elseif nonzeros > nz
            high = gamma
        elseif nonzeros < nz
            low = gamma
        end
    end
    # @assert false
end

rng = MersenneTwister(123)
dim = 5000
points = 5000
X = randn(rng, points, dim);
X .-= mean(X, dims=1)
# S = eTRS.CovarianceMatrix(X)
S = Symmetric(X'*X)

l = 2 # number of previous vectors
Y = randn(rng, dim, l); F = qr(Y); Y = F.Q*Matrix(I, l, l); # Previous vector matrix is orthonormal
nonzeros = 100

@time x, data = binary_search(S, Y, nonzeros)
readline(stdin)
@time x, data = binary_search(S, Y, nonzeros)
# @time x = curve(S, x_init, Y, gamma)
nonzeros = sum(abs.(x) .> 1e-7)
@show nonzeros
x_thresholded = eTRS.get_initial_guess(S, nonzeros; Y=Y)
@show dot(x, S*x) dot(x_thresholded, S*x_thresholded)