__precompile__(true)

module QPnorm

using LinearAlgebra
using Polynomials
using GeneralQP
using TRS 
using DataFrames
using Printf
using SparseArrays
using CSV

include("algebra.jl")
include("qp_norm_boundary.jl")
include("printing.jl")
include("qp_norm.jl")
include("find_feasible_point.jl")
export solve, find_feasible_point

end