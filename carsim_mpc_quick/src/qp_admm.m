function [x,z,y,info] = qp_admm(H,f,A,l,u,x,z,y,opt,R)
%QP_ADMM Toolbox-free OSQP-style ADMM for l <= A*x <= u.
n=size(H,1); m=size(A,1);
if nargin<6 || isempty(x), x=zeros(n,1); end
if nargin<7 || isempty(z), z=zeros(m,1); end
if nargin<8 || isempty(y), y=zeros(m,1); end
rho=opt.rho; sigma=opt.sigma;
if nargin<10 || isempty(R)
    K=(H+sigma*eye(n)+rho*(A'*A));
    [R,p]=chol((K+K')/2);
    assert(p==0,'QP KKT matrix is not positive definite.');
end
primal=inf; dual=inf;
for k=1:opt.max_iter
    x_old=x; z_old=z;
    rhs=-f+sigma*x+rho*A'*(z-y);
    x=R\(R'\rhs);
    Ax=A*x;
    z=min(max(Ax+y,l),u);
    y=y+Ax-z;
    primal=norm(Ax-z,inf);
    dual=norm(rho*A'*(z-z_old)+sigma*(x-x_old),inf);
    eps_p=opt.abs_tol+opt.rel_tol*max(norm(Ax,inf),norm(z,inf));
    eps_d=opt.abs_tol+opt.rel_tol*norm(A'*y,inf);
    if primal<=eps_p && dual<=eps_d, break; end
end
info=struct('iterations',k,'primal_residual',primal, ...
    'dual_residual',dual,'converged',primal<=eps_p && dual<=eps_d);
end
