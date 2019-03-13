__precompile__(true)

module eTRS

using LinearAlgebra, SparseArrays
using Polynomials
include("/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/TRS.jl/src/TRS.jl")
using Main.eTRS.TRS
using SparseArrays
using BenchmarkTools
# using Profile, ProfileView
using Statistics
using DataFrames, CSV

include("algebra.jl")
include("sparse_pca.jl")
include("utils.jl")
include("utils_nonnegative.jl")
include("printing.jl")
export sparce_pca, get_initial_guess, polish!, CovarianceMatrix

end