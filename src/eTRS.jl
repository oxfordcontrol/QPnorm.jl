__precompile__(true)

module eTRS

using LinearAlgebra
using Polynomials

include("/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/GeneralQP.jl/src/GeneralQP.jl")
using Main.eTRS.GeneralQP

include("algebra.jl")
include("extended_trs_boundary.jl")
include("printing.jl")
include("extended_trs.jl")
export solve

end