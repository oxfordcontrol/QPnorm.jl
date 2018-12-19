__precompile__(true)

module eTRS

using LinearAlgebra
using Polynomials

include("algebra.jl")
include("printing.jl")
include("change_constraints.jl")
include("constant_norm_qp.jl")
export solve
# export remove_constraint!, add_constraint!

end