using JLD2
using SparseArrays
include("../src/eTRS.jl")
using Main.eTRS
using Statistics
using Profile, ProfileView

function parallel_get(D, indices)
    D_ = D[:, indices]
    return D_'*D_
end

@load "../examples/docword_nytimes.jld2" D
D = convert(SparseMatrixCSC{Int64,Int64}, D)
m = size(D, 2)
nz = 150
indices = sortperm(randn(2*m))
nz_indices = indices[1:nz]
nz_indices = [2806, 6974, 14404, 14815, 16253, 16259, 17534, 18449, 20214, 22103, 22177, 24491, 25952, 29167, 29175, 29176, 29184, 29191, 29363, 33946, 34784, 35542, 39349, 43808, 43982, 44262, 102918, 103173, 103884, 103986, 104991, 106086, 106261, 106270, 107701, 107933, 107991, 108373, 108729, 108998, 110124, 110128, 110758, 110897, 111199, 111299, 111301, 111328, 112038, 112045, 112341, 112469, 112845, 113597, 114697, 115030, 115042, 115127, 115214, 115481, 116510, 116511, 117266, 117576, 117661, 118213, 118737, 119662, 120000, 120024, 120713, 120860, 122586, 122733, 123167, 123443, 123444, 123614, 123617, 124700, 124751, 126172, 126534, 126607, 127015, 127045, 127537, 128260, 128383, 129064, 129359, 129363, 130364, 130621, 130839, 131069, 131763, 132068, 132084, 132421, 132747, 132750, 132826, 132831, 132958, 133013, 133077, 133222, 133450, 133811, 134168, 134467, 135502, 135753, 136784, 137158, 137513, 139695, 140218, 140459, 140757, 141299, 141678, 141971, 142058, 142272, 142325, 145790, 145792, 145957, 146706, 147704, 147993, 148801, 148819, 154500, 156516, 157412, 158029, 163573, 166815, 171244, 171547, 191247, 195084, 198656, 201391, 201816, 203261, 203832];
# nz_indices = nz_indices[sortperm(randn(nz))]

S = eTRS.CovarianceMatrix(D)
@btime H = eTRS.FlexibleHessian(S, nz_indices);
@btime parallel_get(D, mod.(nz_indices .- 1, m) .+ 1)

#=
Profile.clear()		
Profile.@profile H = eTRS.FlexibleHessian(S, nz_indices);
Profile.clear()		
Profile.@profile H = eTRS.FlexibleHessian(S, nz_indices);
Profile.clear()		
Profile.@profile H = eTRS.FlexibleHessian(S, nz_indices);
ProfileView.view()		
println("Press enter to continue...")
readline(stdin)	
=#