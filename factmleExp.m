

%  This function minimizes the objective function 
%  with out any restriction on S. This menthod is
%  generally good for high dimensions ( dimension >> sample_size ). It does not 
%  need the Covariance matrix S ( dim*dim matrix), due to size restrictions.
%  Instead it uses the data matrix X ( dim*sample_size matrx ) for its calculatons.
%
%  This function calls fval for calculating the objective function.
%
% Inputs
%
%   1. data ----------> A dimension*sample_size matrix containing the data
%   2. diags ---------> Diagonal entris of the covrainece matrix S. 
%   2. rank--------------> The rank constraint on Lambda.
%   3. Psi_init-------> Initial value of Psi.
%   4. tol -----------> Tolrance level
%   5. MAX_ITERS------> Max no of iteration after which programme will
%                       terminate.
%   6. svd_is_true ----> If True uses matlab function svd() to obtain 
%                        singular values and singular vectors. When False, it uses
%                        svds(.) to obtain singular values and singular vectors. 
%                        Please refer to paper https://arxiv.org/abs/1801.05935
%                        for details.
%
%   7. lb ---------------> Lower bound for error variance estimate psi. 
%
%% Stopping criteria
%
% Euclidean norm of (psi_new-psi_old)/psi_old  < threshold 
%                            OR
% The no of iterations > MAX_ITERS.
%
%% Output
%
% hist: 
%     
%              Output Data type-----> 1*4 cell.
%
%    1. hist{1}-------> optimal value of psi.
%    2. hist{2} ------> optimal objective value.
%    3. hist{3} ------> time to calculate each iteration. 
%    4. hist{4}-------> value of objective function at each iteration.

%% CODE

%function [hist] = factmleExp(data,rank,lb,diags,Psi_init,Threshold_l,Threshold_p,MAX_ITERS,svd_is_true) 
function[hist] = factmleExp(data,rank,tol,varargin)

% -------------Preprocessing the input and assigning default values
[~,dim] = size(data);
data = stdd(data); % subtracting mean from data; 

p = inputParser;
p.addRequired('data',@(x) ismatrix(x) ==1 )
p.addRequired('rank',@(x) (x >= 1)&&(x == floor(x) )&&( x <= dim ) )
p.addRequired('tol',@(x) x >= eps )

% Adding optional parameters and default values
p.addParameter('MAX_ITERS',1000,@(x) (x>=1)&&(x == floor(x)));
p.addParameter('lb',10^-3,@(x) (x>eps) );
p.addParameter('svd_is_true',(1>0),@(x) (x == (1>0) )||(x == (1<0)) );
p.addParameter('Psi_init',rand([dim,1]),@(x) iscolumn(x) == 1);
p.addParameter('diags',sum(data.*data)', @(x) (iscolumn(x))&&(length(x) == dim) )

% Parsing the input
p.parse(data,rank,tol,varargin{:})

lb = p.Results.lb; MAX_ITERS = p.Results.MAX_ITERS ; svd_is_true = p.Results.svd_is_true ; Psi_init = p.Results.Psi_init ;
diags = p.Results.diags;


    
    Threshold_l = tol;
    Threshold_p = tol;
    
% --------------------------- end of pre processing --------------------





f= -1*ones(1,MAX_ITERS);
ftime = zeros(1,MAX_ITERS);
f(1)=inf;
Psi = Psi_init;
k=2;
dim=length(Psi);
check=1;

time_start =     tic;

while (check)
    

   Old_Psi=Psi; 
   
% calculating subgradient for mazorization.

x=1./Psi;   % x= Phi [in paper]
x_half=sqrt(x);
s1=bsxfun(@times,x_half',data); % data%Phi^{1/2}

if (svd_is_true==0)
[~,d,U]=svds(s1,rank); 
else
    [~,d,U]=svd(s1,'econ'); d = d(1:rank,1:rank); U = U(:,1:rank);
end



diagd=((diag(d)).^2);

% updating optimal vaue of psi
diff_d= max(0,( 1-1./(diagd)));

Q = bsxfun(@times,1./x_half,U)*(bsxfun(@times,diff_d,U')*(s1')); % This is dim*sample_size  

% diff_psi_0 = sum(Q.*D',2);
diff_psi_0 = sum(Q.*(data'),2); % diag(Q*D) 

Psi=max(diags-diff_psi_0,lb); 

f(k)= calc_fval(diags,x,diagd);

ftime(k) = toc(time_start);

% Convergence criterion
 
if ((f(k)~=inf) && (f(k-1)~=inf))
    
    check= ((  abs(  (  f(k)-f(k-1) )/f(k-1)  )  > Threshold_l)&&(k < MAX_ITERS)) ;
else
    check =((norm((Psi-Old_Psi)./(dim*(Old_Psi)),2) > Threshold_p) && (k < MAX_ITERS));
    
end

k = k+1;


end

hist.Psi = Psi;
hist.Nllopt=f(k-1);
hist.time = ftime(2:(k-1));
hist.Nll = f(2:(k-1));

end

%% fval
function[fval] = calc_fval(diags ,x,eig_val)
% calculates the objective function value.
    fval =  sum(-log(x)) +diags'*x +  sum(    log(max(1,eig_val)) -  max(1,eig_val) +1      );
end

%% stdd
function [ data1] = stdd(data1)
[n,~]=size(data1);
m = mean(data1',2);
data1 = bsxfun(@minus,data1',m)';
data1 = data1/sqrt(n);
end
 
