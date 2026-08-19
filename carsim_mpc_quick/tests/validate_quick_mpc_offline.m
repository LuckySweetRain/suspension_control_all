function result = validate_quick_mpc_offline()
%VALIDATE_QUICK_MPC_OFFLINE Check QP feasibility, preview response, and limits.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'src'));
cfg=quick_mpc_config(); data=build_quick_mpc_data(cfg);
expected_damping=2*cfg.vehicle.corner_damper_Ns_per_m* ...
    [cfg.vehicle.motion_ratio_f^2;cfg.vehicle.motion_ratio_r^2];
assert(max(abs([cfg.vehicle.cf;cfg.vehicle.cr]-expected_damping))<1e-10, ...
    'The prediction-model damping does not match the CarSim passive law.');
expected_effectiveness=cfg.active_scale* ...
    [cfg.vehicle.motion_ratio_f;cfg.vehicle.motion_ratio_r];
assert(max(abs(data.sys.input_effectiveness-expected_effectiveness))<1e-12, ...
    'The prediction input matrix does not match the deployment chain.');
assert(size(data.Acon,1)==3*cfg.horizon*data.sys.nu, ...
    'Force, rate, and travel constraints are not all present in Acon.');
x=zeros(data.sys.nx,1); up=zeros(2,1); warm=[];
N=data.N; dt=cfg.control_dt;
% A bump enters the front preview before the rear axle.
road_t=(0:N)'*dt;
zf=0.035*exp(-((road_t-0.08)/0.025).^2);
delay=(cfg.vehicle.a+cfg.vehicle.b)/20;
zr=0.035*exp(-((road_t-0.08-delay)/0.025).^2);
W=zeros(N*4,1);
for k=1:N
    W((k-1)*4+(1:4))=[zf(k);(zf(k+1)-zf(k))/dt;zr(k);(zr(k+1)-zr(k))/dt];
end
[u,warm,info]=preview_qp_solve(data,x,up,W,warm); %#ok<ASGLU>
assert(all(isfinite(u)),'QP produced non-finite forces.');
assert(all(abs(u)<=cfg.force_limit+1e-6),'Force bound violated.');
assert(all(abs(u-up)<=cfg.force_rate_limit*dt+1e-2),'Rate bound violated.');
assert(info.converged,'QP solver did not converge.');
assert(info.max_constraint_violation<5e-3,'QP constraint violation is excessive.');
assert(info.max_predicted_travel_ratio<=1+5e-3, ...
    'Predicted suspension travel violates its hard constraint.');
result=struct('front_force_N',u(1),'rear_force_N',u(2), ...
    'iterations',info.iterations,'constraint_violation',info.max_constraint_violation, ...
    'max_travel_ratio',info.max_predicted_travel_ratio, ...
    'input_effectiveness',data.sys.input_effectiveness, ...
    'predicted_axle_damping_Ns_per_m',[cfg.vehicle.cf;cfg.vehicle.cr], ...
    'converged',info.converged);
disp(result);
end
