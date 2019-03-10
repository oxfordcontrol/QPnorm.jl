function [x, f] = TPower(A, options, x0)
%%   Truncated power method for sparse eigenvalue problem.
%
%     max x'*A*x    subject to ||x||=1, ||x||_0 <= k.
%
% *** inputs ***
% - A:               p x p symmetric positive semidefinite matrix
% - options:         a structure stores user-specified parameters which include:
%    -- verbose:     level of verbosity (0: no output, 1: final, 2: iter (default), 3: debug
%    -- cardinality: cadinality k (default 10)
%    -- optTol:      optimality tolerance (default: 1e-6)
%    -- maxIter:     max number of iteration (default: 50)
%    -- initType:    initialization type (1: top-1 variance, 2: top-k variance (default))
% - x0:              initialization vector
%
% *** outputs ***
% - x:            p-dimensional sparse eigenvector vector with k non-zeros
% - f:            objective value at the output x
%
% Refer to:
%   Xiao-Tong Yuan, Tong Zhang, Truncated Power Method for Sparse Eigenvalue Problems, Technical Report, 2011
%
% Copyright (C) 2011/2012 - Xiao-Tong Yuan.  

%% Set Parameters
if nargin < 2
    options = [];
end
[verbose, cardinality, optTol, maxIter, initType] = ...
    myProcessOptions(...
    options, 'verbose', 2, 'cardinality', 10, 'optTol', 1e-6, 'maxIter', 50, 'initType', 1);

% Output Parameter Settings
if verbose >= 3
    fprintf('Running TPower Method...\n');
    fprintf('TPower Optimality Tolerance: %.2e\n', optTol);
    fprintf('TPower Maximum Number of Iterations: %d\n', maxIter);
end

%% Output Log
if verbose >= 2
    fprintf('TPower %10s %10s\n', ' Iteration ', ' Objective Val ');
end

%% Default initialization 
if (nargin < 3)
    switch initType
        case 1
            [val,idx]=max(diag(A));
            x0 = zeros(size(A,1),1);
            x0(idx) = 1;
        case 2
    
            [val,idx]=sort(diag(A), 'descend');
            x0 = zeros(size(A,1),1);
            x0(idx(1:cardinality)) = 1;
            x0 = x0 / norm(x0);
    end
end

x = sparse(x0);
% power step
s = A*x;
g = 2*s;
f = x'*s;

% truncate step
x = truncate_operator(g, cardinality);

f_old = f;

i = 1;

%% Main algorithmic loop
while i <= maxIter
    
    % power step
    s = A*x;
    g = 2*s;
    
    % truncate step
    x = truncate_operator(g, cardinality);
    f = x'*s;
    
    if ( abs(f - f_old) < optTol )
        break;
    end
    
    % Output Log
    if verbose >= 2
        fprintf('TPower %10d %10f \n', i, f);
    end
    
    f_old = f;
    i = i+1;
end

end

%% Evaluate the truncate operator
function u = truncate_operator(v , k)

u = zeros(length(v), 1);
[val, idx] = sort(abs(v), 'descend');

v_restrict = v(idx(1:k));
u(idx(1:k)) = v_restrict / norm(v_restrict);

u = sparse(u);
end
