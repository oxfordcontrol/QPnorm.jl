using JLD2
using SparseArrays
include("../src/eTRS.jl")
using Main.eTRS
using CSV
using DataFrames

function generate_principal_vectors(D, vocabulary, nonzeros, k)
    n = size(D, 2)
    L_deflate = zeros(n, 0); R_deflate = zeros(n, 0)
    results = DataFrame()
    for i = 1:5
        S = eTRS.CovarianceMatrix(D, L_deflate, R_deflate)
        x, data = eTRS.binary_search(S, nonzeros)
        nonzero_indices = findall(abs.(x) .> 1e-7)
        results[Symbol("Words_", i)] = vocabulary[:Column1][nonzero_indices]
        results[Symbol("Weights_", i)] = x[nonzero_indices]

        Sx = Vector(S*sparse(x))
        Sx[findall(abs.(x) .< 1e-7)] .= 0.0
        L_deflate = [L_deflate x]
        R_deflate = [R_deflate Sx]
        results |> CSV.write("results.csv")
    end
    return results
end
@load "docword_nytimes.jld2" D
vocabulary = CSV.File("vocab.nytimes.txt", header=0, datarow=1) |> DataFrame
results = generate_principal_vectors(D, vocabulary, 50, 5)
results |> CSV.write("results.csv")