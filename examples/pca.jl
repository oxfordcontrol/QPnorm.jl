using JuMP, Ipopt, Polynomials, LinearAlgebra, Statistics, Arpack
using SparseArrays
using LinearMaps
include("../src/eTRS.jl")
using Main.eTRS

function normalize(x_init, γ)
    x_init ./= norm(x_init)
    while norm(x_init, 0) > γ 
        threshold = minimum(abs.(x_init[abs.(x_init) .> 0]))
        x_init[abs.(x_init) .<= threshold] .= 0.0
        x_init ./= norm(x_init)
    end
    return x_init
end

function pca_ipopt(S, x_init, γ)
    for j = 1:1
        n = length(x_init)
        model = JuMP.Model()
        setsolver(model, IpoptSolver(max_iter=2000))#, print_level=0))
        @variable(model, x[1:n])
        for i in 1:n
        setvalue(x[i], x_init[i]);
        end
        @variable(model, y[1:n])
        for i in 1:n
        setvalue(y[i], abs(x_init[i]));
        end
        @objective(model, Min, -dot(x, S*x)/2)
        @constraint(model, dot(x, x) == 1.0)
        @constraint(model, -y .<= x)
        @constraint(model, x .<= y)
        @constraint(model, sum(y) <= γ)
        status = JuMP.solve(model)
        # @assert status == :Optimal status
        objective = JuMP.getobjectivevalue(model)
        γ *= 1.05
        x_init = getvalue(x)
        return getvalue(x), getvalue(y)
    end
end

function solve_mine(P_, q_, A, b, r, w_init)
    n = Int(length(q_)/2)
    for i = 1:100
        w_init = eTRS.solve_boundary(P_, q_, A, b, r, copy(w_init), verbosity=1, printing_interval=100, max_iter=5000);
        println("γ:", b[end], ", zeros:", sum(abs.(w_init[1:n]) .<= 1e-8))
        b[end] = b[end]*1.25
    end
    return w_init
end

dim = 2000
points = 3000
X = randn(points, dim);
X .-= mean(X, dims=2)
# P = 2*[-A*A' 0*I; 0*I 0*Matrix(I, dim, dim)]
# q = [zeros(dim); ones(dim)]
γ = 10
S = Symmetric(X'*X)

x_init = eigs(S, which=:LR, nev=1)[2][:]
x_init = normalize(x_init, γ)


function mul_p(y, x)
    x1 = view(x, 1:dim); x2 = view(x, dim+1:2*dim)
    y1 = view(y, 1:dim); y2 = view(y, dim+1:2*dim)
    @. y2 = x2 - x1
    mul!(y1, S, y2)
    @. y2 = -x1
    return y
end
eye = SparseMatrixCSC(1.0*I, 2*dim, 2*dim)
P_ = LinearMap(mul_p, 2*dim; ismutating=true, issymmetric=true)
# @show size(P_*randn(2*dim))
# @assert false
# P_ = SparseMatrixCSC([-S S; S -S])
q_ = zeros(2*dim)
A = [-eye; ones(2*dim)']
b = [zeros(2*dim); γ]

w_init = [max.(x_init, 0); -min.(x_init, 0)]
# w_init = [1.0; zeros(n-1); 1.0; zeros(n-1)]
w = eTRS.solve_boundary(P_, q_, A, b, 1.0, copy(w_init), verbosity=1, printing_interval=200, max_iter=5000);

# w = solve_mine(P_, q_, A, b, r, w_init)
@time x_ipopt, y_ipopt = pca_ipopt(X'*X, x_init, γ)
#=
x_ipopt, y_ipopt = pca_ipopt(-S, x_init, γ)
=#
x = w[1:dim] - w[dim+1:2*dim]

@show norm(x)
@show dot(X*x, X*x) dot(X*x_ipopt, X*x_ipopt)

#=
n = dim
model = JuMP.Model()
setsolver(model, IpoptSolver(max_iter=2000))#, print_level=0))
@variable(model, x[1:n])
@variable(model, y[1:n])
for i in 1:n
    setvalue(x[i], 1.0)
    setvalue(y[i], 1.0)
end
@objective(model, Min, -dot(x, S*x))
@constraint(model, dot(x, x) + dot(y, y) == 2.0)
@constraint(model, -y .<= x)
@constraint(model, x .<= y)
@constraint(model, sum(y) <= γ)
status = JuMP.solve(model)
# @assert status == :Optimal status
objective = JuMP.getobjectivevalue(model)
x_2 = getvalue(x)
y_2 = getvalue(y)

show(stdout, "text/plain", [x_1 x_2]); println()
@show norm([x_1; y_1])
@show norm([x_2; y_2])
=#