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
    for i = 1:k
        S = eTRS.CovarianceMatrix(D, L_deflate, R_deflate)
        x, data = eTRS.binary_search(S, nonzeros)
        nonzero_indices = findall(abs.(x) .> 1e-8)
        permutation = sortperm(-abs.(x[nonzero_indices]))
        results[Symbol("Words_", i)] = vocabulary[:Column1][nonzero_indices[permutation]]
        results[Symbol("Weights_", i)] = x[nonzero_indices[permutation]]

        Sx = Vector(S*sparse(x))
        println("Final variance of vector: #", i, " is: ", dot(x, Sx))
        L_deflate = [L_deflate x]
        R_deflate = [R_deflate Sx]
        results |> CSV.write("results.csv")
    end
    return results
end
@load "docword_nytimes.jld2" D
# D = convert(SparseMatrixCSC{Int64,Int64}, D)
D = D./maximum(D);
vocabulary = CSV.File("vocab.nytimes.txt", header=0, datarow=1) |> DataFrame
@time results = generate_principal_vectors(D, vocabulary, 30, 1)
println("Presss enter for next run..."); readline(stdin)
@time results = generate_principal_vectors(D, vocabulary, 30, 1)
nothing
