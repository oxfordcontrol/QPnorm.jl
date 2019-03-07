function varargout=GPower(varargin)
% Generalized power method for sparse principal component analysis.
%
% Z=GPower(A,rho,m,penalty,block,mu)
%
% *** inputs ***  
% A:            (p x n) data matrix
% rho:          sparsity weight factor 
%               (in relative value with respect to the theoretical upper bound)
% m:            number of components
% penalty:      either 'l1' or 'l0'
% block:        either 0 or 1 
%               block==0 means that deflation is used if more than 
%               one component needs to be computed. A block algorithm is 
%               otherwise used, that computes m components at once.
% mu:           vector of dimension m with the mu parameters (required for
%               the block algorithms only)
%
% *** outputs ***
% Z:            (n x m) matrix that contains m sparse loading vectors
%
%
% Refer to:  
%   M. Journée, Y. Nesterov, P. Richtárik, R. Sepulchre, Generalized power 
%   Method for sparse principal component analysis, arXiv:0811.4724v1, 2008
%

A=varargin{1};
RHO=varargin{2};
m=varargin{3};
penalty=varargin{4};
block=varargin{5};
if block==1,
    if nargin>5,
    mu=varargin{6};  % initialization
    else
        disp('Block algorithm: parameters mu_i need to be defined')
        varargout{1}=[];
        return
    end
end

[p,n]=size(A);
iter_max=1000;  % maximum number of admissible iterations
epsilon=1e-4;   % accuracy of the stopping criterion

Z=zeros(n,m);
A_init=A;
%--------------------------------------------------------------------------
if m==1 || (m>1 && block==0),   %single-unit algorithm (deflation is used if m>1)                               
%--------------------------------------------------------------------------
    switch penalty
       case 'l1'        
            for comp=1:m, % loop on the components
                rho=RHO(comp);
                norm_a_i=zeros(n,1);
                for i=1:n,
                    norm_a_i(i)=norm(A(:,i));
                end
                [rho_max,i_max]=max(norm_a_i);
                rho=rho*rho_max;
                x=A(:,i_max)/norm_a_i(i_max); %initialization point  
                f=zeros(iter_max,1); iter=1;
                while 1,
                    Ax=A'*x;
                    tresh=sign(Ax).*max(abs(Ax)-rho,0);
                    f(iter)=sum(tresh.^2); %cost function
                    if f(iter)==0,  %the sparsity parameter is too high: all entries of the loading vector are zero.
                        break                    
                    else
                        %pattern=find(tresh'>0);
                        %grad=A(:,pattern)*(tresh(pattern).*sign(Ax(pattern)));  % gradient
                        grad=A*tresh;
                        x=grad/norm(grad);                        
                    end
                    if iter>2 && (f(iter)-f(iter-1))/f(iter-1)<epsilon || iter>iter_max  % stopping criterion
                        if iter>iter_max, disp('Maximum number of iterations reached'), end
                        break
                    end
                    iter=iter+1;
                end  
                Ax=A'*x;
                pattern=((abs(Ax)-rho) >0); % pattern of sparsity
                z=sign(Ax).*max(abs(Ax)-rho,0);
                if max(abs(z>0))>0, 
                    z=z/norm(z);
                end
                z=pattern_filling(A,pattern,z); % assign values to the nonzero elements
                y=A*z;
                A=A-y*z'; % deflation                
                Z(:,comp)=z;  
            end

    case 'l0'   
        for comp=1:m, % loop on the components  
            rho=RHO(comp);
            norm_a_i=zeros(n,1);
            for i=1:n,
                norm_a_i(i)=norm(A(:,i));
            end
            [rho_max,i_max]=max(norm_a_i);
            rho=rho*rho_max^2;
            x=A(:,i_max)/norm_a_i(i_max); %initialization point                        
            f=zeros(iter_max,1); iter=1;
            while 1,
                Ax=A'*x;
                tresh=max(Ax.^2-rho,0);
                f(iter)=sum(tresh); %cost function
                if  f(iter)==0, %the sparsity parameter is too high: all entries of the loading vector are zero.
                    break                    
                else                   
                    grad=A*((tresh>0).*Ax);
                    x=grad/norm(grad);                   
                end
                if iter>2 && (f(iter)-f(iter-1))/f(iter-1)<epsilon || iter>iter_max % stopping criterion
                    if iter>iter_max, disp('Maximum number of iterations reached'), end
                    break
                end
                iter=iter+1;
            end    
            pattern=((A'*x).^2-rho>0); %pattern of sparsity
            y=x;
            pattern_inv=(pattern==0);
            z=A'*y; z(pattern_inv)=0;
            norm_z=norm(z);
            z=z/norm_z; y=y*norm_z;            
            A=A-y*z'; %deflation
            Z(:,comp)=z;
        end
    end
%--------------------------------------------------------------------------
elseif (m>1 && block && sum(mu-1)==0),  % block algorithm with all paramters mu_i=1
%--------------------------------------------------------------------------
    norm_a_i=zeros(n,1);
    for i=1:n,
        norm_a_i(i)=norm(A(:,i));
    end
    [ignore,i_max]=max(norm_a_i);
    [x,rho_max]=qr([A(:,i_max)/norm_a_i(i_max), randn(p,m-1)],0); %initialization point
    f=zeros(iter_max,1); iter=1;    
    switch penalty
        case 'l1',            
            RHO=RHO*rho_max;
            while 1,
                Ax=A'*x;
                tresh=max(abs(Ax)-repmat(RHO,n,1),0);
                f(iter)=sum(sum(tresh.^2)); %cost function
                if f(iter)==0, %the sparsity parameter is too high: all entries of the loading vector are zero.
                    break
                else
                    grad=zeros(p,m);
                    for i=1:m,
                        pattern=find(tresh(:,i)'>0);
                        grad(:,i)=A(:,pattern)*(tresh(pattern,i).*sign(Ax(pattern,i))); %gradient
                    end
                    [U,S,V]=svd(grad,'econ');
                    x=U*V';                    
                end
                if iter>2 && (f(iter)-f(iter-1))/f(iter-1)<epsilon || iter>iter_max %stopping criterion
                    if iter>iter_max, disp('Maximum number of iterations reached'), end
                    break
                end
                iter=iter+1;
            end
            Ax=A'*x;        
            for i=1:m,
                Z(:,i)=sign(Ax(:,i)).*max(abs(Ax(:,i))-RHO(i),0);
                if max(abs(Z(:,i))>0)>0, 
                    Z(:,i)=Z(:,i)/norm(Z(:,i));
                end
            end     
            pattern=((abs(Ax)-repmat(RHO,n,1)) >0); %pattern of sparsity
            Z=pattern_filling(A,pattern,Z); %assign values to the nonzero elements

        case 'l0',
            RHO=RHO*rho_max^2;
            while 1,                
                Ax=A'*x;
                tresh=max(Ax.^2-repmat(RHO,n,1),0);
                f(iter)=sum(sum(tresh)); %cost function
                if f(iter)==0, %the sparsity parameter is too high: all entries of the loading vector are zero.
                    break
                else
                    grad=zeros(p,m);
                    for i=1:m,
                        pattern=find(tresh(:,i)'>0);
                        grad(:,i)=A(:,pattern)*Ax(pattern,i); %gradient
                    end
                    [U,S,V]=svd(grad,'econ');
                    x=U*V';
                end
                if iter>2 && (f(iter)-f(iter-1))/f(iter-1)<epsilon || iter>iter_max %stopping criterion
                    if iter>iter_max, disp('Maximum number of iterations reached'), end
                    break
                end
                iter=iter+1;
            end
            pattern=((A'*x).^2-repmat(RHO,n,1)>0); %pattern of sparsity
            pattern_inv=(pattern==0);
            Z=A'*x; Z(pattern_inv)=0;
            norm_z=zeros(m,1);
            for i=1:m,
                norm_z(i)=norm(Z(:,i));
                if norm_z(i)>0,
                    Z(:,i)=Z(:,i)/norm_z(i);
                end
            end        
    end  
%--------------------------------------------------------------------------
elseif (m>1 && block && sum(mu-1)~=0),  % block algorithm
%--------------------------------------------------------------------------
    norm_a_i=zeros(n,1);
    for i=1:n,
        norm_a_i(i)=norm(A(:,i));
    end
    [ignore,i_max]=max(norm_a_i);
    [x,rho_max]=qr([A(:,i_max)/norm_a_i(i_max), randn(p,m-1)],0); %initialization point
    f=zeros(iter_max,1); iter=1;
    switch penalty
        case 'l1',     
            RHO=RHO.*mu*rho_max;
            while 1,                
                Ax=A'*x;
                for i=1:m,
                    Ax(:,i)=Ax(:,i)*mu(i);
                end
                tresh=max(abs(Ax)-repmat(RHO,n,1),0);
                f(iter)=sum(sum(tresh.^2)); %cost function
                if f(iter)==0, %the sparsity parameter is too high: all entries of the loading vector are zero.
                    break
                else
                    grad=zeros(p,m);
                    for i=1:m,
                        pattern=find(tresh(:,i)'>0);
                        grad(:,i)=A(:,pattern)*(tresh(pattern,i).*sign(Ax(pattern,i)))*mu(i); %gradient
                    end
                    [U,S,V]=svd(grad,'econ');
                    x=U*V';
                end
                if iter>2 && (f(iter)-f(iter-1))/f(iter-1)<epsilon || iter>iter_max %stopping criterion
                    if iter>iter_max, disp('Maximum number of iterations reached'), end
                    break
                end
                iter=iter+1;
            end
            Ax=A'*x;        
            for i=1:m,
                Ax(:,i)=Ax(:,i)*mu(i);
                Z(:,i)=sign(Ax(:,i)).*max(abs(Ax(:,i))-RHO(i),0);
                if max(abs(Z(:,i))>0)>0, 
                    Z(:,i)=Z(:,i)/norm(Z(:,i));
                end
            end     
            pattern=((abs(Ax)-repmat(RHO,n,1)) >0);  %pattern of sparsity
            Z=pattern_filling(A,pattern,Z,mu); %assign values to the nonzero elements

        case 'l0',
            RHO=RHO.*(mu*rho_max).^2;
            while 1,                
                Ax=A'*x;
                for i=1:m,
                    Ax(:,i)=Ax(:,i)*mu(i);
                end
                tresh=max(Ax.^2-repmat(RHO,n,1),0);
                f(iter)=sum(sum(tresh)); %cost function
                if f(iter)==0, %the sparsity parameter is too high: all entries of the loading vector are zero.
                    break
                else
                    grad=zeros(p,m);
                    for i=1:m,
                        pattern=find(tresh(:,i)'>0);
                        grad(:,i)=A(:,pattern)*Ax(pattern,i)*mu(i); %gradient 
                    end                      
                    [U,S,V]=svd(grad,'econ');
                    x=U*V';
                end
                if iter>2 && (f(iter)-f(iter-1))/f(iter-1)<epsilon || iter>iter_max %stopping criterion
                    if iter>iter_max, disp('Maximum number of iterations reached'), end
                    break
                end
                iter=iter+1;
            end
            Ax=A'*x;
            for i=1:m,
                Ax(:,i)=Ax(:,i)*mu(i);
            end
            pattern=((Ax).^2-repmat(RHO,n,1)>0); %pattern of sparsity
            pattern_inv=(pattern==0);
            Z=A'*x; Z(pattern_inv)=0;
            norm_z=zeros(m,1);
            for i=1:m,
                norm_z(i)=norm(Z(:,i));
                if norm_z(i)>0,
                    Z(:,i)=Z(:,i)/norm_z(i);
                end
            end       
    end
end
varargout{1}=Z;

%==========================================================================
%==========================================================================
% Subfunction
%==========================================================================
function varargout=pattern_filling(varargin)
% Compute a local maximizer of
% max_{X,Z} trace(X^T A Z N)  s.t. X^T X=I_m and Z(P)=0 and Diag(Z^T Z)=I_m
%

A=varargin{1};      % data matrix
P=varargin{2};      % pattern of sparsity
if nargin>2,
    Z=varargin{3};  % initialization
end
if nargin>3,
    mu=varargin{4};
end
[p,n]=size(A);
m=size(P,2);

%--------------------------------------------------------------------------
if m==1,     % single-unit case
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
        A_red=A(:,support);
        while 1
            tmp=A_red'*(A_red*u);
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
    varargout{1}=z;
    
%--------------------------------------------------------------------------    
elseif nargin==3,  % block case with all mu_i equal to 1
    pattern_inv=P==0;    
    iter_max=1000;
    epsilon=1e-6;
    f=zeros(iter_max,1); iter=1;
    while 1
        AZ=A*Z;              
        [U,S,V]=svd(AZ,'econ');
        X=U*V';
        ff=0;
        for i=1:m,
            ff=ff+X(:,i)'*AZ(:,i);
        end  
        f(iter)=ff;
        Z=A'*X;
        Z(pattern_inv)=0;
        for i=1:m,
            norm_Z=norm(Z(:,i));
            if norm_Z>0,
                Z(:,i)=Z(:,i)/norm_Z;
            end
        end         
        if iter>2 && (abs(f(iter)-f(iter-1)))/abs(f(iter-1))<epsilon || iter>iter_max,
            if iter>iter_max, disp('Maximum number of iterations reached'), end
            break
        end
        iter=iter+1;
    end    
    varargout{1}=Z;
%--------------------------------------------------------------------------    
else                   % block case
    pattern_inv=P==0;    
    iter_max=1000;
    epsilon=1e-6;
    f=zeros(iter_max,1); iter=1;
    while 1
        AZ=A*Z;
        for i=1:m,
            AZ(:,i)=AZ(:,i)*mu(i);
        end                
        [U,S,V]=svd(AZ,'econ');
        X=U*V';
        ff=0;
        for i=1:m,
            ff=ff+X(:,i)'*AZ(:,i);
        end  
        f(iter)=ff;
        Z=A'*X;
        for i=1:m,
            Z(:,i)=Z(:,i)*mu(i);
        end  
        Z(pattern_inv)=0;
        for i=1:m,
            norm_Z=norm(Z(:,i));
            if norm_Z>0,
                Z(:,i)=Z(:,i)/norm_Z;
            end
        end             
        if iter>2 && (abs(f(iter)-f(iter-1)))/abs(f(iter-1))<epsilon || iter>iter_max,
            if iter>iter_max, disp('Maximum number of iterations reached'), end
            break
        end
        iter=iter+1;
    end    
    varargout{1}=Z;
end
