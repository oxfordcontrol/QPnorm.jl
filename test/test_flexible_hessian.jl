include("../src/eTRS.jl")
using Main.eTRS
using Random, Test
using Statistics

rng = MersenneTwister(123)
n = 100 # Number of observations
m = 50 # Number of dimensions
nz = 10 # Nonzeros
D = randn(rng, n, m)
# D .-= mean(D, dims=1)
indices = sortperm(randn(m))
nz_indices = indices[1:nz]
S = eTRS.CovarianceMatrix(D)
# D_ = D .- mean(D, dims=1)
# show(stdout, "text/plain", (D_'*D_)[nz_indices, nz_indices]); println()
H = eTRS.FlexibleHessian(S, nz_indices)
x = zeros(2*m)
x[nz_indices] = randn(nz)

@testset "Flexible Hessian - Create" begin
    x_ = x[1:m] - x[m+1:end]
    @test abs(x_'*(S*x_) + x[nz_indices]'*(H.H*x[nz_indices])) <= 1e-7
end

eTRS.add_column!(H, indices[nz + 1])
append!(nz_indices, indices[nz + 1])
nz += 1
x .= 0
x[nz_indices] = randn(nz)
@testset "Flexible Hessian - Add Column" begin
    x_ = x[1:m] - x[m+1:end]
    @test abs(x_'*(S*x_) + x[nz_indices]'*(H.H*x[nz_indices])) <= 1e-7
end

idx = Int(floor(nz/2))
eTRS.remove_column!(H, idx)
deleteat!(nz_indices, idx)
nz -= 1
x .= 0
x[nz_indices] = randn(nz)
@testset "Flexible Hessian - Remove Column" begin
    x_ = x[1:m] - x[m+1:end]
    @test abs(x_'*(S*x_) + x[nz_indices]'*(H.H*x[nz_indices])) <= 1e-7
end