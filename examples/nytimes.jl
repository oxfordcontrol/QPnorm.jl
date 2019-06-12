using JLD2
using SparseArrays
using LinearAlgebra
include("../src/QPnorm.jl")
using Main.QPnorm
using CSV
using DataFrames

function generate_principal_vectors(D, vocabulary, nonzeros, k, nonnegative=false)
    # k is the number of the requested principal vectors
    n = size(D, 2)
    L_deflate = zeros(n, 0); R_deflate = zeros(n, 0)
    results = DataFrame()
    for i = 1:k
        S = QPnorm.CovarianceMatrix(D, L_deflate, R_deflate)
        if !nonnegative
            x, data, t = QPnorm.binary_search(S, nonzeros)
        else
            x, data, t = QPnorm.binary_search_nonnegative(S, nonzeros)
        end
        println("Time spent in the active-set algorithm for PC #", i, " : ", t, " seconds.")
        nonzero_indices = findall(abs.(x) .> 1e-8)
        permutation = sortperm(-abs.(x[nonzero_indices]))
        results[Symbol("Words_", i)] = vocabulary[:Column1][nonzero_indices[permutation]]
        results[Symbol("Weights_", i)] = x[nonzero_indices[permutation]]

        Sx = Vector(S*sparse(x))
        println("Final variance of vector: #", i, " is: ", dot(x, Sx))
        L_deflate = [L_deflate x]
        R_deflate = [R_deflate Sx]
        if !nonnegative
            results |> CSV.write("results.csv")
        else
            results |> CSV.write("results_nonnegative.csv")
        end
    end
    return results
end
@load "docword_nytimes.jld2" D
println("----Running standard sparse pca----")
D = convert(SparseMatrixCSC{Int64,Int64}, D)
D = D./maximum(D);
vocabulary = CSV.File("vocab.nytimes.txt", header=0, datarow=1) |> DataFrame
results = generate_principal_vectors(D, vocabulary, 30, 1)
println("Presss enter for next run..."); readline(stdin)
results = generate_principal_vectors(D, vocabulary, 30, 1)

println("Presss enter to produce nonnegative PCA results") readline(stdin)
println("----Running nonnegative sparse pca----")
results_nonnegative = generate_principal_vectors(D, vocabulary, 30, 5, true)
println("Presss enter for next run..."); readline(stdin)
results_nonnegative = generate_principal_vectors(D, vocabulary, 30, 5, true)
nothing
