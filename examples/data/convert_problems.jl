using CUTEst
using Arpack
using NLPModels
using SparseArrays
using JLD2
using JuMP
using Gurobi
include("classification.jl")

function convert_problems(dataset)
    working_dir = pwd()

    if dataset == "MASTSIF"
        dir = ENV["MASTSIF"]
        savedir = string(working_dir, "./")
        @assert false "Dataset name must be MASTSIF"
    end
    cd(dir)
    parameters = CSV.File(string(savedir, "parameters.csv")) |> DataFrame

    if isfile(string(savedir, "statistics.csv"))
        @warn "Loading previous statistics.csv file. Delete it if you don't want this to happen."
        df = CSV.File(string(savedir, "statistics.csv")) |> DataFrame
    else
        df = DataFrame(name=String[], n = Int[], m = Int[],
                hessian_nonzeros = Int[], constraints_nonzeros = Int[])
    end

    try
        files = sort(filter_problems(savedir))
        start_idx = 184
        files = files[start_idx:end]
        counter = start_idx
        for file in files
            println("Touching file #", counter)
            counter += 1
            # Any parameters passed in CUTEstModel (except file) are passed to sifdecoder
            # see http://www.cuter.rl.ac.uk/sifdec/Doc/general/node18.html for an explanation
            idx = findfirst(parameters[:Name] .== file[1:end-4])
            nlp = nothing; x0 = zeros(10000)
            if idx != nothing && parameters[:Parameter][idx] != "-"
                println("Calling: CUTEstModel(", file, " -param ", parameters[:Parameter][idx], ")")
                nlp = CUTEstModel(file, "-param", parameters[:Parameter][idx]) # "-param", "N=1000")
                x0 = nlp.meta.x0
            elseif idx == nothing
                println("Calling: CUTEstModel(", file, ")")
                nlp = CUTEstModel(file) # , "-param", "N=1000")
                x0 = nlp.meta.x0
            end

            try
                n = length(x0)
                if n <= 2000
                    A_ = NLPModels.jac(nlp, zeros(size(x0)))
                    b_ = -NLPModels.cons(nlp, zeros(size(x0)))
                    A_[nlp.meta.jlow, :] .*= -1
                    b_[nlp.meta.jlow] .*= -1
                    l = nlp.meta.lvar
                    u = nlp.meta.uvar
                    # Now constraints are A_x <= b_ and l <= x <= u
                    eye = SparseMatrixCSC(1.0*I, n, n)
                    A = [A_;
                        eye[isfinite.(u), :];
                        -eye[isfinite.(l), :]
                        ]
                    b = [b_;
                        u[isfinite.(u)];
                        -l[isfinite.(l)]
                        ]
                    m, n = size(A)
                    # Now constraints are Ax <= b
                    x0 = find_closest_feasible_point(A, b, x0)
                    # Objective: 0.5x'Px + q'x + f0
                    P = NLPModels.hess(nlp, x0); P = (P + P'); P = P - spdiagm(0 => Vector(diag(P)./2));
                    q = NLPModels.grad(nlp, x0) - P*x0 
                    f0 = NLPModels.obj(nlp, x0) - (1/2*dot(x0, P*x0) - dot(q, x0))
                    # If the objective is quadratic it might be more accurate to calculate q, x0 as
                    # q = NLPModels.grad(nlp, zeros(size(x0)))
                    # f0 = NLPModels.obj(nlp, zeros(size(x0)))
                    println("Checking convexity for: ", file, ", size of P: ", size(P))
                    #=
                    # Sanity check for objective
                    x = randn(n)
                    relative_obj_error = norm(NLPModels.obj(nlp, x) - (0.5*dot(x, P*x) + dot(q, x) + f0))/(NLPModels.obj(nlp, x) + 1e-12)
                    @assert relative_obj_error <= 1e-10 relative_obj_error NLPModels.obj(nlp, x)
                    =#
                    λ_min = minimum(eigvals(Matrix(P)))
                    #= Indirect computation of smallest eigenvalue
                    λ_min = 0.0
                    try 
                        λ_min = eigs(Symmetric(P), nev=1, which=:SR, ritzvec=false)[1]
                    catch
                        @warn "Using direct solver to calculate smallest eigenvalue"
                        λ_min = minimum(eigvals(Matrix(P)))
                    end
                    =#
                    if λ_min <= -1e-7
                        println("Accepted file:", file, " minimum_eigevalue:", λ_min)
                        push!(df, [file[1:end-4], n, m, nnz(P), nnz(A)])
                        # @show df
                        @assert sum(isnan.(P)) == 0 && sum(isnan.(q)) == 0 && sum(isnan.(A)) == 0 && sum(isnan.(b)) == 0
                        @save string(savedir, file[1:end-4], ".jld2") P q A b x0 # f0 
                        df |> CSV.write(string(savedir, "statistics.csv"))
                    end
                else
                    println("Ignoring: ", file[1:end-4], " due to size.")
                end
            catch
                println("Failed: ", file[1:end-4])
            finally
                finalize(nlp)
            end
        end
    finally
        cd(working_dir)
    end
end

function find_closest_feasible_point(A, b, x0)
    m, n = size(A)

    model = JuMP.Model()
    setsolver(model, GurobiSolver(OutputFlag=0))
    @variable(model, x[1:n])
    @constraint(model, A*x .<= b)
    @objective(model, Min, dot(x - x0, x - x0))
    status = JuMP.solve(model)
    if status != :Optimal
        @warn "Gurobi could not solve initial problem"
        @assert false
    end
    return getvalue(x)
end