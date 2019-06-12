## Reproducing the numerical examples from the [paper](https://arxiv.org/abs/1906.04022)
You can reproduce the numerical examples of the Sparse PCA part of the paper as following.
* First run `sh download_nytimes_docword.sh` to download the data.
* Then run `julia docword_to_sparse_matrix.jl` to convert the data to a proper format.
* Finally, run `nytimes.jl` 
The results are written in the file `results.csv` and `results_nonnegative.csv`.

`GPower` and `TPower` results are obtained by first converting the data to `MATLAB` format by running `docword_to_sparse_matrix.m` and then running `nytimes_gpower.m` and `nytimes_tpower.m` in the `Matlab Code` folder.