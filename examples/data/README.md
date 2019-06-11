# Saving second order approximations of the CUTEst library to .jld2 files

This README shows how to convert problems from the [`CUTEr/st` `"MASTSIF"`](ftp.numerical.rl.ac.uk/pub/cutest/sif/mastsif.html) collection to problems of the form
```
minimize    ½x'Px + q'x + f0
subject to  Ax ≤ b
```
and stores the matrices `P, q, A, b` as well as `f0` and a suggested starting point `x0` in `.jld2` files in this folder (`data`).

If the respective `.sif` problem is not a (convex/non-convex) QP then `P, q` correspond to a second order approximation of the (non-linear) cost and `A, b` to linearisation of the (potentially non-linear) constraints, at the initial point `x0` provided in the `.sif` file.

Furthermore, the file `statistics.csv` includes statistics about each converted problem.

## Prerequisites
The following Julia packages are required
* `JuliaSmoothOptimizers/CUTEst.jl`
* `DataFrames.jl`
* `CSV.jl`
* `NLPModels`
* `Gurobi.jl`

##  How to use
```
include("convert_problems.jl")
convert_problems("MASTSIF")
```
Sometimes `convert_problems()` `SEGFAULTS` with e.g. the following output
```
OpenBLAS : Program will terminate because you tried to start too many threads.
[1]    16632 segmentation fault  /Applications/Julia-1.0.app/Contents/Resources/julia/bin/julia run_me.jl
```
This might be due to a problem in the [underlying library](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) used. In this case, set `start_idx` to the index of the last successful file and call again `convert_problems()` to continue the conversion.