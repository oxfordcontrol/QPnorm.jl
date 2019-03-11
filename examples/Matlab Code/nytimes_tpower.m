load docword_nytimes.mat
vocabulary = readtable("vocab.nytimes.txt");

D = D./max(max(D));
means = mean(D, 1)';

x = zeros(0);
nz = 50; % Desired nonzeros
% init_type = "default_tpower";
init_type = "truncutated_eigenvector";

tic;

l_deflate = zeros(size(D, 1), 0);
r_deflate = zeros(size(D, 2), 0);
options.cardinality = nz;
for k = 1:1 % Number of sparse principal vectors desired
    % Define function handles for deflated, centered dataset
    A = @(x) mul(D, means, l_deflate, r_deflate, x);
    At = @(x) mul_t(D, means, l_deflate, r_deflate, x);
    S = @(x) At(full(A(x)));% + 30*x;
        
    if init_type == "default_tpower"
        % Default initialization in TPower.
        diag_S = zeros(size(D, 2), 1);
        for i = 1:length(diag_S)
            diag_S(i) = norm(D(:, i))^2 + length(diag_S)*means(i)^2 - 2*sum(D(:, i))*means(i); % WARNING: no deflation
        end
        [val,idx]=max(diag_S);
        x0 = zeros(size(D,2),1);
        x0(idx) = 1;
    elseif init_type == "truncutated_eigenvector"
        % Initialize with truncutated max eigenvector
        [x0, l] = eigs(S, size(D, 2), 1, 'lr', 'tol', 1e-6);
        x0 = truncate_operator(x0, options.cardinality);
        pattern=((abs(x0) > 1e-9)); % pattern of sparsity
        D_red = D(:, pattern);
        A_red = @(x) mul(D_red, means(pattern), l_deflate, r_deflate(pattern, :), x);
        At_red = @(x) mul_t(D_red, means(pattern), l_deflate, r_deflate(pattern, :), x);
        x0=pattern_filling(A_red, At_red, pattern, x0);
    else
        assert false
    end
    
    % x0 = randn(size(D, 2), 1);
    z = TPower(S, options, x0);
    indices = find(abs(z(:)) > 1e-7);
    % Print resulting vocabulary
    [vocabulary(indices, 1) table(z(indices))]

    y = A(z);
    l_deflate = [l_deflate y];
    r_deflate = [r_deflate z];
end
toc

function y = mul(A, mu, l_deflate, r_deflate, x)
    y = A*x - mu'*x;
    for i = 1:size(l_deflate, 2)
        y = y - l_deflate(:, i)*(r_deflate(:, i)'*x);
    end
end

function y = mul_t(A, mu, l_deflate, r_deflate, x)
    y = A'*x - sum(x)*mu;
    for i = 1:size(l_deflate, 2)
        y = y - r_deflate(:, i)*(l_deflate(:, i)'*x);
    end
end

function u = truncate_operator(v , k)
    u = zeros(length(v), 1);
    [val, idx] = sort(abs(v), 'descend');

    v_restrict = v(idx(1:k));
    u(idx(1:k)) = v_restrict / norm(v_restrict);
end