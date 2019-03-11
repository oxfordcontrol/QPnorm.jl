load docword_nytimes.mat
vocabulary = readtable("vocab.nytimes.txt");

D = D./max(max(D));
means = mean(D, 1)';

x = zeros(0);
nz = 50; % Desired nonzeros

tic;

l_deflate = zeros(size(D, 1), 0);
r_deflate = zeros(size(D, 2), 0);
for k = 1:1 % Number of sparse principal vectors desired
    % Define function handles for deflated, centered dataset
    A = @(x) mul(D, means, l_deflate, r_deflate, x);
    At = @(x) mul_t(D, means, l_deflate, r_deflate, x);
    S = @(x) At(full(A(x)));
    % Get initial point
    % [x, rho_max] = get_initial_point(D, means, l_deflate, r_deflate);
    rho_max = 500;
    [x, l] = eigs(S, size(D, 2), 1, 'lr', 'tol', 1e-6);
    x = (D*x); x = x/norm(x);
    % x = randn(size(D, 1), 1); rho_max = 50; % Dummy start
    
    high = 0.2;
    low = 0.00001;
    for i = 1:50 % Maximum iterations of binary search
        gamma = (high - low)/2 + low;
        
        rho=gamma*rho_max;
        tic;
        x=GPower(A, At, x, rho);
        Ax=At(x);
        pattern=((abs(Ax)-rho) >0); % pattern of sparsity
        z=sign(Ax).*max(abs(Ax)-rho,0);
        if max(abs(z>0))>0
            z=z/norm(z);
        end
        D_red = D(:, pattern);
        A_red = @(x) mul(D_red, means(pattern), l_deflate, r_deflate(pattern, :), x);
        At_red = @(x) mul_t(D_red, means(pattern), l_deflate, r_deflate(pattern, :), x);
        z=pattern_filling(A_red, At_red, pattern, z); % assign values to the nonzero elements
        toc;
        
        nonzeros = nnz(z)
        gamma
        if nonzeros == nz
            display("Found solution at iteration: "); i
            break
        elseif nonzeros > nz
            low = gamma;
        elseif nonzeros < nz
            high = gamma;
        end
    end
    y = A(z);
    l_deflate = [l_deflate y];
    r_deflate = [r_deflate z];
    
    indices = find(abs(z(:)) > 1e-7);
    [vocabulary(indices, 1) table(z(indices))]
end
toc;


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

function [x, rho_max] = get_initial_point(D, means, l_deflate, r_deflate)
    n = size(D, 2);
    norm_A_i=zeros(n,1);
    for i=1:n,
        a = D(:, i);
        a = a - means(i)*ones(size(a)); % Remove mean
        for j = 1:size(l_deflate, 2) % Deflate
            a = a - l_deflate(:, j)*r_deflate(i, j);
        end
        norm_A_i(i)=norm(a);
    end
    [rho_max,i_max]=max(norm_A_i);

    a_max = D(:, i_max);
    a_max = a_max - means(i_max)*ones(size(a_max)); % Remove mean
    for j = 1:size(l_deflate, 2) % Deflate
        a_max = a_max - l_deflate(:, j)*r_deflate(i_max, j);
    end
    x=a_max/norm_A_i(i_max); %initialization point  
end