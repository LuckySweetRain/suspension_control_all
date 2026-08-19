function [u_cmd,warm,info] = preview_qp_solve(data,x0,u_prev,W,warm,W_bound)
%PREVIEW_QP_SOLVE Solve one constrained preview MPC problem.
cfg=data.cfg; N=data.N; nu=data.sys.nu;
assert(numel(x0)==data.sys.nx && numel(W)==N*data.sys.nw);
if nargin<5 || isempty(warm)
    warm=struct('x',repmat(u_prev./cfg.force_limit,N,1),'z',[],'y',[]);
end
if nargin<6 || isempty(W_bound), W_bound=zeros(size(W)); end

c=data.Yx*x0+data.Yw*W;
dc=zeros(N*nu,1); dc(1:nu)=-u_prev;
dc_scaled=dc./data.dmax;
f=2*(data.MyScaled'*data.Qy*c+data.Dscaled'*data.Rdu*dc_scaled);

travel_affine=(data.Tx*x0+data.Tw*W)./data.travel_limit;
travel_margin=zeros(size(travel_affine));
if cfg.robust.enabled
    travel_margin=(abs(data.Tw)*abs(W_bound))./data.travel_limit;
    travel_margin=min(travel_margin,cfg.robust.max_travel_tightening_ratio);
end
hold_bound=zeros(size(data.Ahold,1),1);
l=[-ones(N*nu,1);-ones(N*nu,1)-dc_scaled; ...
    -ones(N*nu,1)+travel_margin-travel_affine;hold_bound];
u=[ ones(N*nu,1); ones(N*nu,1)-dc_scaled; ...
     ones(N*nu,1)-travel_margin-travel_affine;hold_bound];

if isempty(warm.z), warm.z=zeros(size(data.Acon,1),1); end
if isempty(warm.y), warm.y=zeros(size(data.Acon,1),1); end
[sol,warm.z,warm.y,solver]=qp_admm(data.H,f,data.Acon,l,u, ...
    warm.x,warm.z,warm.y,cfg.admm,data.qp_chol);
warm.x=[sol(nu+1:end);sol(end-nu+1:end)];
u_cmd=data.force_scale(1:nu).*sol(1:nu);
u_cmd=min(max(u_cmd,-cfg.force_limit),cfg.force_limit);
du_limit=cfg.force_rate_limit*cfg.control_dt;
u_cmd=min(max(u_cmd,u_prev-du_limit),u_prev+du_limit);
info=solver;
info.max_constraint_violation=max([0;data.Acon*sol-u;l-data.Acon*sol]);
info.objective=0.5*sol'*data.H*sol+f'*sol;
info.max_predicted_travel_ratio=max(abs(travel_affine+data.Tscaled*sol));
info.max_robust_travel_margin=max(travel_margin);
info.solution=sol;
end
