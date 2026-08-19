function data = build_quick_mpc_data(cfg)
%BUILD_QUICK_MPC_DATA Condense the half-car finite-horizon QP and road data.
if nargin < 1, cfg = quick_mpc_config(); end
if ~isfolder(cfg.generated_dir), mkdir(cfg.generated_dir); end
sys = half_car_linear_model(cfg);
N=sys.nu*0+cfg.horizon; nx=sys.nx; nu=sys.nu; nw=sys.nw;

Sx=zeros(N*nx,nx); Su=zeros(N*nx,N*nu); Sw=zeros(N*nx,N*nw);
for i=1:N
    ri=(i-1)*nx+(1:nx);
    Sx(ri,:)=sys.Ad^i;
    for j=1:i
        cu=(j-1)*nu+(1:nu); cw=(j-1)*nw+(1:nw);
        power=sys.Ad^(i-j);
        Su(ri,cu)=power*sys.Bd;
        Sw(ri,cw)=power*sys.Ed;
    end
end

Cy=kron(eye(N),sys.Cy); Dy=kron(eye(N),sys.Dy); Fy=kron(eye(N),sys.Fy);
My=Cy*Su+Dy;
Yx=Cy*Sx;
Yw=Cy*Sw+Fy;
wy=cfg.weights.output./(cfg.weights.output_scale.^2);
Qy=kron(eye(N),diag(wy));
D=eye(N*nu);
for i=2:N
    r=(i-1)*nu+(1:nu); c=(i-2)*nu+(1:nu);
    D(r,c)=-eye(nu);
end
% Optimize dimensionless force v, with U = force_scale .* v. Scaling is
% essential because force is O(1e3 N) while suspension travel is O(1e-2 m).
force_scale=repmat(cfg.force_limit,N,1);
Sforce=diag(force_scale);
dmax=repmat(cfg.force_rate_limit*cfg.control_dt,N,1);
travel_limit=repmat(cfg.suspension_limit,N,1);
MyScaled=My*Sforce;
Dscaled=diag(1./dmax)*D*Sforce;
Ru=cfg.weights.force*eye(N*nu);
Rdu=cfg.weights.force_rate*eye(N*nu);

% Hard bounds: normalized force, force rate, predicted suspension travel,
% and optional first-move blocking for persistent/CDC deployment.
Ctr=sys.Cy(3:4,:);
Tmat=kron(eye(N),Ctr)*Su;
Tx=kron(eye(N),Ctr)*Sx;
Tw=kron(eye(N),Ctr)*Sw;
Tscaled=diag(1./travel_limit)*Tmat*Sforce;
% The travel rows are state- and preview-dependent. Their affine bounds are
% assembled online in preview_qp_solve.
hold_steps=1;
Ahold=zeros(0,N*nu);
if ~strcmpi(cfg.deployment.mode,'ideal_force')
    hold_steps=min(N,max(1,round( ...
        cfg.deployment.minimum_dwell_s/cfg.control_dt)));
    Ahold=zeros((hold_steps-1)*nu,N*nu);
    for i=2:hold_steps
        rows=(i-2)*nu+(1:nu);
        Ahold(rows,1:nu)=-eye(nu);
        Ahold(rows,(i-1)*nu+(1:nu))=eye(nu);
    end
end
Acon=[eye(N*nu);Dscaled;Tscaled;Ahold];
H=2*(MyScaled'*Qy*MyScaled+Ru+Dscaled'*Rdu*Dscaled);
H=(H+H')/2+1e-8*eye(size(H));
K=H+cfg.admm.sigma*eye(size(H))+cfg.admm.rho*(Acon'*Acon);
[qp_chol,p_chol]=chol((K+K')/2);
assert(p_chol==0,'QP KKT matrix is not positive definite.');

[road_station,road_elevation]=extract_carsim_road_table(cfg.carsim_run_par);
road_sensor=build_spatial_road_sensor(cfg,road_station,road_elevation);
data=struct('cfg',cfg,'sys',sys,'N',N,'Sx',Sx,'Su',Su,'Sw',Sw, ...
    'My',My,'MyScaled',MyScaled,'Yx',Yx,'Yw',Yw,'Qy',Qy, ...
    'D',D,'Dscaled',Dscaled,'Rdu',Rdu,'H',H,'force_scale',force_scale, ...
    'dmax',dmax,'travel_limit',travel_limit,'Acon',Acon,'qp_chol',qp_chol, ...
    'Tmat',Tmat, ...
    'Tscaled',Tscaled,'Tx',Tx,'Tw',Tw,'Ahold',Ahold, ...
    'hold_steps',hold_steps, ...
    'road_station',road_station, ...
    'road_elevation',road_elevation, ...
    'road_sensor',road_sensor);
save(cfg.data_file,'data');
end
