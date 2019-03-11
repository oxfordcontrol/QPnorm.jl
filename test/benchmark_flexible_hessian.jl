using JLD2
using SparseArrays
include("../src/eTRS.jl")
using Main.eTRS
using Statistics
using Profile, ProfileView

@load "../examples/docword_nytimes.jld2" D
D = convert(SparseMatrixCSC{Int64,Int64}, D)
m = size(D, 2)
nz = 50
indices = sortperm(randn(2*m))
nz_indices = indices[1:nz]

S = eTRS.CovarianceMatrix(D)
@btime H = eTRS.FlexibleHessian(S, nz_indices);
H = eTRS.FlexibleHessian(S, nz_indices);
nothing
#=
Profile.clear()		
Profile.@profile eTRS.FlexibleHessian(S, nz_indices);
Profile.clear()		
Profile.@profile eTRS.FlexibleHessian(S, nz_indices);
Profile.clear()		
Profile.@profile eTRS.FlexibleHessian(S, nz_indices);
ProfileView.view()		
println("Press enter to continue...")
readline(stdin)	
=#