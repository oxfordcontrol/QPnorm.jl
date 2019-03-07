dims = 15000;
observations = 300;
X = randn(observations, dims);
X = X - mean(X, 1);

x = zeros(0);
nz = 500; % Desired nonzeros
high = 0.2;
low = 0.00001;

tic;
for i = 1:50
    gamma = (high - low)/2 + low;
    % x = zeros(0); % Uncomment to disable warm starting
    [Z, x] = GPower(X, gamma, 1, 'l1', 0, x);
    nonzeros = nnz(Z)
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
toc;