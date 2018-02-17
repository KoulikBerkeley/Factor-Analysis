% Generating the data

exp_mean = 10; L_mean = 10; L_var = 1;
rank_pop = 8; pp = 200; N = 2200;

d =  diag(random('exp',exp_mean,[pp,1]));
l =random('norm',L_mean,L_var,[pp,rank_pop]);
 
Sigma=d + l*(l');
eig_is_true = (1>0);

RR=chol(Sigma);
dimension = size(Sigma,1);

data = randn(N,dimension)*RR; S=cov(data,1);

curr_rank = 5; tol = 10^-5;

% Using Factmle with default options
y = factmleExp(data,curr_rank,tol);

% Using optional parameters. 
maxit = 2000; Psi_init = rand([dimension,1]);

y1 = factmle(S,curr_rank,tol,'MAX_ITERS',maxit);
y2 = factmle(S,curr_rank,tol,'MAX_ITERS',maxit,'Psi_init',Psi_init);
y3 = factmle(S,curr_rank,tol,'eig_is_true',(1<0));



% Large dimension 
clear all

exp_mean = 1; L_mean = 10; L_var = 1;
rank_pop = 5; pp = 10000; N = 50;

d =  diag(random('exp',exp_mean,[pp,1]));
l =random('norm',L_mean,L_var,[pp,rank_pop]);
 
Sigma=d + l*(l');

RR=chol(Sigma);
dimension = size(Sigma,1);

data = randn(N,dimension)*RR; 
curr_rank = 5; tol = 10^-5;

% factmleExp with default options
x = factmleExp(data,curr_rank,tol);

% Using optional parameters. 
maxit = 2000; Psi_init = rand([dimension,1]);

x1 = factmleExp(data,curr_rank,tol,'MAX_ITERS',maxit);
x2 = factmleExp(data,curr_rank,tol,'MAX_ITERS',maxit,'Psi_init',Psi_init);
x3 = factmleExp(data,curr_rank,tol,'svd_is_true',(1<0));








