function prepare_orin_data()
%PREPARE_ORIN_DATA Generate a dedicated deployment artifact and parity fixture.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'src'));
cfg=quick_mpc_config();
% The 12 s CarSim sweep needs 1103 iterations near t=9.14 s; the original
% 800-iteration cap stops before feasibility. Keep this deployment override
% separate from the research baseline. This mode is simulation-time lockstep.
cfg.admm.max_iter=3200;
cfg.generated_dir=fullfile(fileparts(root),'results','orin');
cfg.data_file=fullfile(cfg.generated_dir,'quick_mpc_data.mat');
data=build_quick_mpc_data(cfg);
% Exercise initialization, moving states, preview, warm start and nonzero force.
n=80; exports=zeros(39,n); reference=zeros(8,n); diagnostics=zeros(10,n);
states=zeros(8,n); previews=zeros(4*data.N,n); bounds=previews;
est=[]; warm=[]; previous=zeros(2,1);
for k=1:n
    t=(k-1)*cfg.control_dt;
    y=zeros(39,1); y(1)=0.6+0.001*sin(4*t); y(2)=0.0144*cos(4*t);
    y(4)=0.01*sin(3*t); y(5)=0.03*cos(3*t);
    y(7:10)=20+sin(5*t)*[2;3;2;3];
    y(11:14)=cos(5*t)*[10;15;10;15];
    y(23:26)=cos(5*t)*[10;15;10;15];
    y(31:34)=[10;10-cfg.vehicle.a-cfg.vehicle.b;10;10-cfg.vehicle.a-cfg.vehicle.b]+t*35/3.6;
    y(39)=35;
    [x,est]=carsim_half_car_state(y,est,cfg);
    [W,meta]=carsim_road_preview(y,data);
    [u,warm,info]=preview_qp_solve(data,x,previous,W,warm,meta.W_bound);
    previous=u;
    total=cfg.carsim_damper_gain*y(23:26)+0.5*cfg.active_scale*u([1;2;1;2]);
    total=min(max(total,-cfg.corner_force_limit),cfg.corner_force_limit);
    exports(:,k)=y; reference(:,k)=[total;total];
    diagnostics(:,k)=[u;info.iterations;info.max_constraint_violation; ...
        double(info.converged);meta.used_distance_m;meta.required_distance_m; ...
        1;meta.coverage_fraction;info.max_robust_travel_margin];
    states(:,k)=x; previews(:,k)=W; bounds(:,k)=meta.W_bound;
end
save(fullfile(cfg.generated_dir,'parity_fixture.mat'), ...
    'exports','reference','diagnostics','states','previews','bounds');
fprintf('Prepared %s\n',cfg.data_file);
end
