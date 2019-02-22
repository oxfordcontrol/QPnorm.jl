using JuMP, Ipopt, Polynomials, LinearAlgebra, Statistics, Arpack
using SparseArrays
using LinearMaps
include("../src/eTRS.jl")
using Main.eTRS
using Random

rng = MersenneTwister(123)
dim = 10000
points = 300
X = sprand(rng, points, dim, 0.7);
X .-= mean(X, dims=1)
S = eTRS.CovarianceMatrix(X)# Symmetric(X'*X)

l = 3 # number of previous vectors
Y = randn(rng, dim, l); F = qr(Y); Y = F.Q*Matrix(I, l, l); # Previous vector matrix is orthonormal
x_init, gamma = eTRS.get_initial_guess(S, 50; Y=Y)

initial_nonzeros = sum(abs.(x_init) .> 1e-7)
@show initial_nonzeros
x = eTRS.sparse_pca(S, gamma, x_init; Y=Y, verbosity=1, printing_interval=200, max_iter=10000);
nonzeros = sum(abs.(x) .> 1e-7)
@show nonzeros
x_thresholded, gamma = eTRS.get_initial_guess(S, nonzeros; Y=Y)
eTRS.polish!(x, S, Y=Y)
@show dot(x, S*x) dot(x_thresholded, S*x_thresholded)

function pca_ipopt_implicit(D::Matrix{T}, x_init::Vector{T}, γ::T) where {T}
    n = length(x_init)
    model = JuMP.Model()
    setsolver(model, IpoptSolver(max_iter=2000))#, print_level=0))
    @variable(model, x[1:n])
    for i in 1:n
        setvalue(x[i], x_init[i]);
    end
    @variable(model, y[1:n])
    for i in 1:n
        setvalue(y[i], abs(x_init[i]));
    end
    z_init = D*x_init
    @variable(model, z[1:size(D, 1)])
    for i in 1:length(z_init)
        setvalue(z[i], z_init[i]);
    end
    @objective(model, Min, -dot(z, z)/2)
    @constraint(model, z .== D*x)
    @constraint(model, dot(x, x) <= 1.0)
    @constraint(model, -y .<= x)
    @constraint(model, x .<= y)
    @constraint(model, sum(y) <= γ)
    status = JuMP.solve(model)
    # @assert status == :Optimal status
    objective = JuMP.getobjectivevalue(model)
    return getvalue(x), getvalue(y)
end

function pca_ipopt(S::Matrix{T}, x_init::Vector{T}, γ::T) where {T}
    n = length(x_init)
    model = JuMP.Model()
    setsolver(model, IpoptSolver(max_iter=2000))#, print_level=0))
    @variable(model, x[1:n])
    for i in 1:n
        setvalue(x[i], x_init[i]);
    end
    @variable(model, y[1:n])
    for i in 1:n
        setvalue(y[i], abs(x_init[i]));
    end
    @objective(model, Min, -dot(x, S*x)/2)
    @constraint(model, dot(x, x) <= 1.0)
    @constraint(model, -y .<= x)
    @constraint(model, x .<= y)
    @constraint(model, sum(y) <= γ)
    status = JuMP.solve(model)
    # @assert status == :Optimal status
    objective = JuMP.getobjectivevalue(model)
    return getvalue(x), getvalue(y)
end

# pca_ipopt(S.data, x_init, gamma)
# pca_ipopt_implicit(X, x_init, gamma)