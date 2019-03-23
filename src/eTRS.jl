__precompile__(true)

module eTRS

using LinearAlgebra
using Polynomials
using GeneralQP
using TRS
using Main.eTRS.TRS
using DataFrames
using Printf
using CSV

include("algebra.jl")
include("extended_trs_boundary.jl")
include("printing.jl")
include("extended_trs.jl")
export solve

end