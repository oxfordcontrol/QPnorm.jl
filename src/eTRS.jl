__precompile__(true)

module eTRS

using LinearAlgebra, SparseArrays
using Polynomials
using GeneralQP
# kinclude("/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/GeneralQP.jl/src/GeneralQP.jl")
using Main.eTRS.GeneralQP
using SparseArrays
using BenchmarkTools
# using Profile, ProfileView
using Pardiso
using Statistics
using DataFrames, CSV

include("algebra.jl")
include("extended_trs_boundary.jl")
include("printing.jl")
include("extended_trs.jl")
export solve

end