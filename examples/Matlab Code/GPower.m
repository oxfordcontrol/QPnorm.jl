function x=GPower(A, At, x, rho)
    % Generalized power method for sparse principal component analysis.
    %
    % Z=GPower(A,rho,m,penalty,block,mu)
    %
    % *** inputs ***  
    % A:            data matrix function handle
    % x:       initial guess (especially useful for warm starting)
    % rho:          sparsity weight factor 
    %
    % *** outputs ***
    % z:            the sparse principal vector
    % x
    %
    %
    % Refer to:  
    %   M. Journée, Y. Nesterov, P. Richtárik, R. Sepulchre, Generalized power 
    %   Method for sparse principal component analysis, arXiv:0811.4724v1, 2008
    %

    n=size(x);
    iter_max=1000;  % maximum number of admissible iterations
    epsilon=1e-4;   % accuracy of the stopping criterion

    f=zeros(iter_max,1); iter=1;
    while 1,
        Ax=At(x);
        tresh=sign(Ax).*max(abs(Ax)-rho,0);
        f(iter)=sum(tresh.^2); %cost function
        if f(iter)==0,  %the sparsity parameter is too high: all entries of the loading vector are zero.
            break                    
        else
            %pattern=find(tresh'>0);
            %grad=A(:,pattern)*(tresh(pattern).*sign(Ax(pattern)));  % gradient
            grad=A(tresh);
            % norm(x - grad/norm(grad))
            % iter
            x=grad/norm(grad);                        
        end
        if iter>2 && (f(iter)-f(iter-1))/f(iter-1)<epsilon || iter>iter_max  % stopping criterion
            if iter>iter_max, disp('Maximum number of iterations reached'), end
            break
        end
        iter=iter+1;
    end
    %f(iter-10+2:iter)-f(iter-10+1:iter-1)
    %semilogy(f(2:end)-f(1:end-1))
end