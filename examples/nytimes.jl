using JLD2
using SparseArrays
include("../src/eTRS.jl")
using Main.eTRS

@load "docword_nytimes.jld2" D
S = eTRS.CovarianceMatrix(D)
nonzeros = 50
x, data = eTRS.binary_search(S, nonzeros)

# x_thresholded = eTRS.get_initial_guess(S, 50)
