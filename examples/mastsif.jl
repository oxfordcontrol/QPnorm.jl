using LinearAlgebra, Random
using BenchmarkTools
using JLD2, FileIO
using JuMP
using Glob
include("../src/eTRS.jl")
include("./subproblems.jl")
using Main.eTRS
using DataFrames
using CSV
using SparseArrays
# using Profile, ProfileView

working_dir = pwd()
path = "/Users/nrontsis/OneDrive - The University of Oxford/PhD/Code/CUTEst.jl/data/MASTSIF/"
cd(path)
files = glob("*.jld2")
cd(working_dir)

df = DataFrame(name=String[], n = Int[], m = Int[],
    time = Float64[], time_ipopt = Float64[],
    objective = Float64[], objective_ipopt = Float64[],
    infeasibility = Float64[], infeasibility_ipopt = Float64[],
    solution = [], solution_ipopt = [])

for file in files
    file = string("NCVXQP1", ".jld2")
    filepath = string(path, file)
    q = load(filepath, "q")
    P, q, A, b, x0 = load(filepath, "P", "q", "A", "b", "x0")
    # We want to solve
    # minimize    ½x'Px + q'x + f0
    # subject to  Ax ≤ b
    #             ‖x - x0‖ = r
    # We convert the problem to a standard form by a change of variables z = x - x0
    q = q + P*x0
    b = b - A*x0
    # f0 += dot(data.x0, (P*data.x0))/2 + dot(q, x0)
    x_init = zeros(size(x0))
    # @show maximum(A*x_init - b)
    m, n = size(A)
    @show m, n
    
    println("Solving problem: ", file[1:end-5], " with norm(x_init)=", norm(x_init))
    r = 100.0
    x_ipopt = solve_ipopt(P, q, A, b, r, x_init)
    @show @elapsed x_ipopt = solve_ipopt(P, q, A, b, r, x_init)
    eTRS.solve(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=1, max_iter=1300);
    eTRS.solve(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=500, max_iter=1300);
    return

    #=
    Profile.clear()		
    Profile.@profile eTRS.solve(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=500, max_iter=1000);# return nothing
    Profile.clear()		
    Profile.@profile eTRS.solve(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=500, max_iter=1000);# return nothing
    ProfileView.view()		
    println("Press enter to continue...")		
    readline(stdin)		
    Profile.clear()		
    Profile.@profile eTRS.solve(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=500, max_iter=200);# return nothing
    Profile.clear()		
    Profile.@profile eTRS.solve(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=500, max_iter=200);# return nothing
    ProfileView.view()		
    readline(stdin)		
    println("Press enter to continue...")		
    return
    =#
    if true # file != "HIMMELBJ.jld2"
        x_ipopt = Float64[]; f_ipopt = NaN; infeasibility_ipopt = NaN; t_ipopt = NaN
        try
            x_ipopt = solve_ipopt(P, q, A, b, r, x_init)
            t_ipopt = @elapsed x_ipopt = solve_ipopt(P, q, A, b, r, x_init)
            f_ipopt = dot(x_ipopt, P*x_ipopt)/2 + dot(q, x_ipopt)
            infeasibility_ipopt = max(maximum(A*x_ipopt - b), norm(x_ipopt) - r, 0)
        catch e
            nothing
        end

        x = Float64[]; f = NaN; infeasibility = NaN; t = NaN
        P = Matrix(P); A = Matrix(A)
        try
            x = eTRS.solve(P, q, A, b, r, copy(x_init), verbosity=1, printing_interval=500, max_iter=5000)
            t = @elapsed x = eTRS.solve(P, q, A, b, r, x_init, verbosity=1, printing_interval=500, max_iter=5000)
            f = dot(x, P*x)/2 + dot(x, q)
            infeasibility = max(maximum(A*x - b), norm(x) - r, 0)
        catch e
            nothing
        end

        push!(df, [file[1:end-5], n, m,
                t, t_ipopt,
                f, f_ipopt,
                infeasibility, infeasibility_ipopt,
                x, x_ipopt])
        
        df |> CSV.write(string("statistics.csv"))
    end
end