using JuMP, Ipopt, Polynomials, LinearAlgebra, Statistics, Arpack
using SparseArrays
using LinearMaps
include("../src/QPnorm.jl")
using Main.QPnorm
using Random

rng = MersenneTwister(123)
dim = 5000
points = 500
X = randn(rng, points, dim);
# X .-= mean(X, dims=1)
S = QPnorm.CovarianceMatrix(X)
# S = X'*X

l = 2 # number of previous vectors
Y = randn(rng, dim, l); F = qr(Y); Y = F.Q*Matrix(I, l, l); # Previous vector matrix is orthonormal
nonzeros = 100

@time x, data = QPnorm.binary_search(S, nonzeros, Y)
readline(stdin)
@time x, data = QPnorm.binary_search(S, nonzeros, Y)
# @time x = curve(S, x_init, Y, gamma)
nonzeros = sum(abs.(x) .> 1e-7)
@show nonzeros
x_thresholded = QPnorm.get_initial_guess(S, nonzeros; Y=Y)
@show dot(x, S*x) dot(x_thresholded, S*x_thresholded)