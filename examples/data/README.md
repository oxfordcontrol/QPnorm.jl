# Saving second order approximations of the CUTEst library to .jld2 files

This README shows how to convert problems from the [`CUTEr/st` `"MASTSIF"`](ftp.numerical.rl.ac.uk/pub/cutest/sif/mastsif.html) collection to problems of the form
```
minimize    ½x'Px + q'x + f0
subject to  Ax ≤ b
```
and stores the matrices `P, q, A, b` as well as `f0` and a suggested starting point `x0` in `.jld2` files in this folder (`data`).

If the respective `.sif` problem is not a (convex/non-convex) QP then `P, q` correspond to a second order approximation of the (non-linear) cost and `A, b` to linearisation of the (potentially non-linear) constraints, at the initial point `x0` provided in the `.sif` file.

Parameters can be passed to the problems by including the respective name and the desired parameter on the file `parameters.csv`.

Furthermore, the file `statistics.csv` includes statistics about each converted problem.

Finally, as described in the paper, we skip problems with number of variables more than 2000, or problems that are convex.

## Prerequisites
The following Julia packages are required
* `JuliaSmoothOptimizers/CUTEst.jl 0.4.0`
* `DataFrames.jl 0.17.0`
* `CSV.jl 0.4.3`
* `NLPModels.jl 0.8.0`
* `Gurobi.jl 0.5.9`
* `JuMP.jl 0.19.0`

##  How to use
Run in Julia:
```julia
include("convert_problems.jl")
convert_problems("MASTSIF")
```
Sometimes `convert_problems()` `SEGFAULTS` with e.g. the following output
```
OpenBLAS : Program will terminate because you tried to start too many threads.
[1]    16632 segmentation fault  /Applications/Julia-1.0.app/Contents/Resources/julia/bin/julia run_me.jl
```
This might be due to a problem in the [underlying library](https://github.com/JuliaSmoothOptimizers/CUTEst.jl) used. In this case, set `start_idx` inside `convert_problems()` to the index of the last successful file and call again `convert_problems()` to continue the conversion.