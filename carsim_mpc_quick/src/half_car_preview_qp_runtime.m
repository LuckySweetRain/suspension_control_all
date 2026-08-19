function [corner_forces,diagnostics] = half_car_preview_qp_runtime(command,y)
%HALF_CAR_PREVIEW_QP_RUNTIME Persistent runtime used by Simulink and tests.
persistent data estimator_state u_prev warm last_info actuator
if nargin<1, command='step'; end
if strcmpi(command,'reset')
    data=[]; estimator_state=[]; u_prev=zeros(2,1); warm=[]; last_info=[];
    actuator=[]; corner_forces=zeros(8,1); diagnostics=zeros(10,1); return
end
if isempty(data)
    cfg=quick_mpc_config();
    if isfile(cfg.data_file), s=load(cfg.data_file,'data'); data=s.data;
    else, data=build_quick_mpc_data(cfg); end
    u_prev=zeros(2,1);
    actuator=initialize_actuator(data.cfg);
    last_info=empty_solver_info();
end
y=double(y(:));
if numel(y)==77
    y=y(39:77); % second, complete CarSim export set
elseif numel(y)~=39
    corner_forces=zeros(8,1); diagnostics=[0;0;0;inf;0;0;0;0;0;0]; return
end
if any(~isfinite(y))
    corner_forces=zeros(8,1); diagnostics=[0;0;0;inf;0;0;0;0;0;0]; return
end

[x,estimator_state]=carsim_half_car_state(y,estimator_state,data.cfg);
[W,preview]=carsim_road_preview(y,data);
cfg=data.cfg;
actuator.sample_index=actuator.sample_index+1;
decision_steps=max(1,round(cfg.deployment.decision_period_s/cfg.control_dt));
decision_due=strcmpi(cfg.deployment.mode,'ideal_force') || ...
    mod(actuator.sample_index-1,decision_steps)==0;

if cfg.active_scale<=eps
    requested=zeros(2,1);
    applied_u=zeros(2,1);
    last_info=empty_solver_info();
else
    if decision_due
        [requested,warm,last_info]=preview_qp_solve( ...
            data,x,u_prev,W,warm,preview.W_bound);
        actuator.requested=requested;
    else
        requested=actuator.requested;
    end
    [applied_u,actuator]=deploy_command(requested,y,actuator,cfg,decision_due);
end
u_prev=applied_u;

baseline=cfg.carsim_damper_gain*y([23 24 25 26]);
if cfg.active_scale<=eps
    total4=baseline;
elseif strcmpi(cfg.deployment.mode,'cdc')
    damping4=actuator.applied_damping([1 2 1 2]);
    total4=damping4.*y([23 24 25 26]);
else
    active=cfg.active_scale*[0.5*applied_u(1);0.5*applied_u(2); ...
        0.5*applied_u(1);0.5*applied_u(2)];
    total4=baseline+active;
end
total4=min(max(total4,-cfg.corner_force_limit),cfg.corner_force_limit);
corner_forces=[total4;total4]; % duplicated imports in the current Run_all.par
diagnostics=[applied_u;last_info.iterations; ...
    last_info.max_constraint_violation;double(last_info.converged); ...
    preview.used_distance_m;preview.required_distance_m; ...
    double(decision_due);preview.coverage_fraction; ...
    last_info.max_robust_travel_margin];
end

function actuator=initialize_actuator(cfg)
actuator=struct( ...
    'sample_index',0, ...
    'requested',zeros(2,1), ...
    'target_force',zeros(2,1), ...
    'applied_force',zeros(2,1), ...
    'target_damping',cfg.carsim_damper_gain*ones(2,1), ...
    'applied_damping',cfg.carsim_damper_gain*ones(2,1), ...
    'last_change_sample',-inf);
end

function info=empty_solver_info()
info=struct('iterations',0,'max_constraint_violation',0,'converged',true, ...
    'max_robust_travel_margin',0);
end

function [applied_u,state]=deploy_command(requested,y,state,cfg,decision_due)
mode=lower(string(cfg.deployment.mode));
if mode=="ideal_force"
    state.target_force=requested;
    state.applied_force=requested;
    applied_u=requested;
    return
end

dwell_steps=max(1,round(cfg.deployment.minimum_dwell_s/cfg.control_dt));
may_change=state.sample_index-state.last_change_sample>=dwell_steps;
beta=response_fraction(cfg.control_dt,cfg.deployment.response_time_constant_s);
if mode=="persistent_force"
    if decision_due && may_change
        quant=cfg.deployment.force_quantization_N;
        target=quant*round(requested/quant);
        target=min(max(target,-cfg.force_limit),cfg.force_limit);
        if any(abs(target-state.target_force)>eps)
            state.target_force=target;
            state.last_change_sample=state.sample_index;
        end
    end
    state.applied_force=state.applied_force+ ...
        beta*(state.target_force-state.applied_force);
    applied_u=state.applied_force;
    return
end

assert(mode=="cdc",'Unknown deployment mode: %s',cfg.deployment.mode);
velocity4=y([23 24 25 26]);
velocity_axle=[mean(velocity4([1 3]));mean(velocity4([2 4]))];
if decision_due && may_change
    desired=state.target_damping;
    threshold=cfg.deployment.cdc_velocity_threshold_mmps;
    moving=abs(velocity_axle)>=threshold;
    desired(moving)=cfg.carsim_damper_gain+ ...
        0.5*cfg.active_scale*requested(moving)./velocity_axle(moving);
    levels=cfg.deployment.cdc_damping_levels_N_per_mmps(:);
    target=state.target_damping;
    for i=1:2
        [~,index]=min(abs(levels-desired(i)));
        target(i)=levels(index);
    end
    if any(abs(target-state.target_damping)>eps)
        state.target_damping=target;
        state.last_change_sample=state.sample_index;
    end
end
state.applied_damping=state.applied_damping+ ...
    beta*(state.target_damping-state.applied_damping);
active_axis_force=2*(state.applied_damping-cfg.carsim_damper_gain).*velocity_axle;
applied_u=active_axis_force/max(cfg.active_scale,eps);
applied_u=min(max(applied_u,-cfg.force_limit),cfg.force_limit);
end

function beta=response_fraction(dt,tau)
if tau<=eps, beta=1; else, beta=1-exp(-dt/tau); end
end
