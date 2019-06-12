## Reproducing the numerical examples from the [paper](https://arxiv.org/abs/1906.04022)
You can reproduce the numerical examples by running the following two files:
* `random_constant_norm.jl` Random QPs that are constrained to have constant norm
* `cutest.jl` Compute SQP directions for the nonconvex problems `CUTEst` with linear constraints and less than 2000 variables. In order to run this file you need to first convert the `CUTEst` problems to a proper format, which can be done with the `convert_problems.jl` file in the `data` folder.

Check the branch `pca` for the sparse pca results.