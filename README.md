# QPnorm.jl: Performing Sparse-PCA with an active-set algorithm
This repository is the official `Julia` implementation for the Sparse-PCA part of the [following paper](https://arxiv.org/abs/1906.04022):
```
Rontsis N., Goulart P.J., & Nakatsukasa, Y.
An active-set algorithm for norm constrained quadratic problems
Preprint in Arxiv
```
i.e. an active-set algorithm for solving problems the sparse-pca problem
```
maximize    ½x'Σx
subject to  ‖x‖_2 ≤ 1
            ‖x‖_1 ≤ γ
```
where `x` in the `n-`dimensional variable `Σ` is the `n x n` covariance matrix of the data and `γ > 1` is a parameter that controls the sparsity.

Although the core algorithm is the same as the main branch, this branch is designed to take advantage of the special structure of the Sparse-PCA problem.

## Installation
This repository is based on [TRS.jl](https://github.com/oxfordcontrol/TRS.jl) and [GeneralQP.jl](https://github.com/oxfordcontrol/GeneralQP.jl). First install these two dependencies by running 
```
add https://github.com/oxfordcontrol/TRS.jl#0a9b641
add https://github.com/oxfordcontrol/GeneralQP.jl#4c74666
```
in [Julia's Pkg REPL mode](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html#Getting-Started-1) and then install this repository by running
```
add https://github.com/oxfordcontrol/QPnorm.jl#pca
```

## Reproducing the numerical examples from the [paper](https://arxiv.org/abs/1906.04022)
The code for reproducing the Sparse-PCA part of the numerical examples from the paper is in the folder `examples`.