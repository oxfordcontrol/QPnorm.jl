__precompile__(true)

module eTRS

using LinearAlgebra
using Polynomials

include("algebra.jl")
include("extended_trs_boundary.jl")
include("printing.jl")
include("extended_trs.jl")
export solve

end