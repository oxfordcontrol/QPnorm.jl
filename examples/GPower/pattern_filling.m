function z=pattern_filling(A_red, At_red, P, Z)
% Compute a local maximizer of
% max_{X,Z} trace(X^T A Z N)  s.t. X^T X=I_m and Z(P)=0 and Diag(Z^T Z)=I_m
%

n=size(Z, 1);

support=find(P);
if isempty(support),
    z_red=zeros(n,1);
    support=1:n;      
elseif length(support)==1,
    z_red=1;
else
    u=Z(support);
    epsilon=1e-6; iter_max=1000;
    iter=1; f=zeros(iter_max,1); 
    while 1
        tmp=At_red(A_red(u));
        u=tmp/norm(tmp);
        f(iter)=-2*u'*tmp; 
        if iter>2 && abs(f(iter)-f(iter-1))/abs(f(iter-1))<epsilon || iter>iter_max,
            if iter>iter_max, disp('Max. number of iterations'), end
            break
        end
        iter=iter+1;
    end
    z_red=u;
end
z=zeros(n,1);
z(support)=z_red;
end % Pattern-filling