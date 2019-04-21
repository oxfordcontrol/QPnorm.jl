using Ipopt
using SparseArrays
function solve_ipopt(P::AbstractMatrix, q, A::AbstractMatrix, b,
    r_max=Inf, r_min=-Inf, x_init=zeros(0);
    max_iter=5000, print_level=0, bound_relax_factor=NaN)

    n = size(P, 1)
    A_ = [A; ones(n)']
    m = size(A_, 1)

    eval_f(x) = dot(x, Symmetric(P)*x)/2 + dot(q, x)
    eval_grad_f(x, grad_f) = grad_f[:] = P*x + q

    function eval_g(x, g)
        g[1:end-1] = A*x - b
        g[end] = dot(x, x)
    end

    function eval_jac_g(x, mode, rows, cols, values)
        if mode == :Structure
            k = 1
            for j = 1:n, i=1:m rows[k] = i
                cols[k] = j
                k += 1
            end
        else
            A_[end, :] .= 2*x
            values[:] .= reshape(A_, length(A_))
        end
    end

    function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
        if mode == :Structure
            k = 1
            @inbounds for j = 1:n, i=j:n
                rows[k] = i; cols[k] = j
                k += 1
            end
        else
            k = 1
            @inbounds for j=1:n, i=j:n
                values[k] = obj_factor*P[j, i]
                if i == j
                    values[k] += 2*lambda[end]
                end
                k +=1 
            end
        end
    end

    x_L = -Inf*ones(n)
    x_U = Inf*ones(n)

    g_L = -Inf*ones(m)
    g_L[end] = r_min^2
    g_U = zeros(m)
    g_U[end] = r_max^2

    nz_P = Int(n*(n + 1)/2)
    nz_A_ = m*n

    prob = createProblem(n, x_L, x_U, m, g_L, g_U, nz_A_, nz_P,
                eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
    addOption(prob, "max_iter", max_iter)
    addOption(prob, "print_level", print_level)
    if !isnan(bound_relax_factor)
        addOption(prob, "bound_relax_factor", bound_relax_factor)
    end
    if length(x_init) == n
        prob.x .= x_init
    end

    timing = @elapsed status = solveProblem(prob)
    # println(Ipopt.ApplicationReturnStatus[status])
    # println(prob.obj_val)
    prob.mult_g[end] *= 2
    return prob.x, prob.mult_g, timing
end

function solve_ipopt(P::SparseMatrixCSC, q, A::SparseMatrixCSC, b,
    r_max=Inf, r_min=-Inf, x_init=zeros(0);
    max_iter=5000, print_level=0, bound_relax_factor=NaN)

    # Ensure diagonal is present
    n = size(P, 1)
    P = P + 1e-60*I
    P_lower = SparseMatrixCSC(LowerTriangular(P))
    A_ = [A; ones(1, n)]
    m, n = size(A_)

    eval_f(x) = dot(x, P*x)/2 + dot(q, x)
    eval_grad_f(x, grad_f) = grad_f[:] = P*x + q

    function eval_g(x, g)
        # Bad: g    = zeros(2)  # Allocates new array
        # OK:  g[:] = zeros(2)  # Modifies 'in place'
        g[1:end-1] = A*x - b
        g[end] = dot(x, x)
    end


    function eval_jac_g(x, mode, rows, cols, values)
        if mode == :Structure
            rows[:], cols[:] = findnz(A_)
        else
            m, n = size(A_)
            @inbounds for i = 1:n
                A_[end, i] = 2*x[i]
            end
            values[:] .= A_.nzval
        end
    end

    function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
        if mode == :Structure
            rows[:], cols[:] = findnz(P_lower)
        else
            P_ = obj_factor*P_lower + 2*lambda[end]*I
            values[:] .= P_.nzval
        end
    end

    x_L = -Inf*ones(n)
    x_U = Inf*ones(n)

    g_L = -Inf*ones(m)
    g_L[end] = r_min^2
    g_U = zeros(m)
    g_U[end] = r_max^2

    nz_P = nnz(P_lower)
    nz_A_ = nnz(A_)

    prob = createProblem(n, x_L, x_U, m, g_L, g_U, nz_A_, nz_P,
                eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
    addOption(prob, "max_iter", max_iter)
    addOption(prob, "print_level", print_level)
    if !isnan(bound_relax_factor)
        addOption(prob, "bound_relax_factor", bound_relax_factor)
    end
    if length(x_init) == n
        prob.x .= x_init
    end
    timing = @elapsed status = solveProblem(prob)

    # println(Ipopt.ApplicationReturnStatus[status])
    # println(prob.obj_val)
    # println(prob.mult_g)
    prob.mult_g[end] *= 2
    return prob.x, prob.mult_g, timing
end
