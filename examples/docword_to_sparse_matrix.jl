using CSV
using DataFrames
using JLD2
using SparseArrays

df = CSV.File("docword.nytimes.txt", header=0, datarow=4, delim=" ") |> DataFrame
I = convert(Vector{Int32}, df[:Column1])
J = convert(Vector{Int32}, df[:Column2])
V = convert(Vector{Int16}, df[:Column3])
D = sparse(I, J, V)
# @save "docword_nytimes.jld2" D
