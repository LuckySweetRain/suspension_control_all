function tuning = tune_quick_mpc_offline()
%TUNE_QUICK_MPC_OFFLINE Select weights and compare preview horizons.
% The reduced model is used only for parameter screening. Final claims must
% be based on the CarSim closed-loop rerun.
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root, fullfile(root,'src'));
cfg0 = quick_mpc_config();
roads = define_offline_roads(cfg0);

weight_sets = define_weight_sets(cfg0);
passive_c = simulate_case(cfg0, roads.class_c, false, 8.0);
weight_rows = repmat(empty_result_row(), numel(weight_sets), 1);
for i = 1:numel(weight_sets)
    cfg = apply_weights(cfg0, weight_sets(i));
    active = simulate_case(cfg, roads.class_c, true, 8.0);
    score = normalized_score(active, passive_c);
    weight_rows(i) = result_row(weight_sets(i).name, cfg.horizon, ...
        roads.class_c.id, active, passive_c, score);
end
weight_table = struct2table(weight_rows);
[~, best_idx] = min(weight_table.Score);
best_weights = weight_sets(best_idx);

horizons = [10 15 20 30];
horizon_rows = repmat(empty_result_row(), numel(horizons)*numel(roads.list), 1);
row_idx = 0;
for i = 1:numel(horizons)
    cfg = apply_weights(cfg0, best_weights);
    cfg.horizon = horizons(i);
    for j = 1:numel(roads.list)
        road = roads.list(j);
        stop_time = road.stop_time;
        passive = simulate_case(cfg, road, false, stop_time);
        active = simulate_case(cfg, road, true, stop_time);
        row_idx = row_idx + 1;
        horizon_rows(row_idx) = result_row(best_weights.name, horizons(i), ...
            road.id, active, passive, normalized_score(active,passive));
    end
end
horizon_table = struct2table(horizon_rows);
mean_score = groupsummary(horizon_table,'Horizon','mean','Score');
minimum_score = min(mean_score.mean_Score);
near_optimal = mean_score.mean_Score <= 1.02*minimum_score;
best_horizon = min(mean_score.Horizon(near_optimal));

out_dir = fullfile(cfg0.results_dir,'offline_tuning');
if ~isfolder(out_dir), mkdir(out_dir); end
writetable(weight_table,fullfile(out_dir,'weight_search.csv'));
writetable(horizon_table,fullfile(out_dir,'horizon_sweep.csv'));
writetable(mean_score,fullfile(out_dir,'horizon_mean_score.csv'));
make_horizon_figure(horizon_table,fullfile(out_dir,'horizon_sweep.png'));

tuning = struct('best_weights',best_weights,'best_horizon',best_horizon, ...
    'weight_table',weight_table,'horizon_table',horizon_table, ...
    'mean_score',mean_score);
save(fullfile(out_dir,'offline_tuning.mat'),'tuning');
disp(weight_table);
disp(mean_score);
fprintf('Recommended weight set: %s\n',best_weights.name);
fprintf('Recommended horizon: N=%d (%.2f s)\n',best_horizon, ...
    best_horizon*cfg0.control_dt);
end

function sets = define_weight_sets(cfg)
sets(1) = weight_set('comfort', ...
    [1.40;0.90;0.12;0.12;0.03;0.03], ...
    cfg.weights.output_scale,1e-3,5e-3);
sets(2) = weight_set('balanced', ...
    [1.00;0.60;0.18;0.18;0.04;0.04], ...
    cfg.weights.output_scale,2e-3,1e-2);
sets(3) = weight_set('travel_aware', ...
    [0.90;0.55;0.35;0.35;0.05;0.05], ...
    cfg.weights.output_scale,2e-3,1e-2);
sets(4) = weight_set('smooth_control', ...
    [1.00;0.60;0.18;0.18;0.04;0.04], ...
    cfg.weights.output_scale,1e-2,1e-1);
sets(5) = weight_set('very_smooth', ...
    [1.00;0.60;0.18;0.18;0.04;0.04], ...
    cfg.weights.output_scale,2e-2,3e-1);
sets(6) = weight_set('conservative', ...
    [1.00;0.60;0.18;0.18;0.04;0.04], ...
    cfg.weights.output_scale,5e-2,6e-1);
sets(7) = weight_set('actuator_margin', ...
    [1.00;0.60;0.18;0.18;0.04;0.04], ...
    cfg.weights.output_scale,2e-1,6e-1);
sets(8) = weight_set('low_effort', ...
    [1.00;0.60;0.18;0.18;0.04;0.04], ...
    cfg.weights.output_scale,5e-1,1.0);
sets(9) = weight_set('headroom', ...
    [1.00;0.60;0.18;0.18;0.04;0.04], ...
    cfg.weights.output_scale,1.0,1.0);
sets(10) = weight_set('low_force', ...
    [1.00;0.60;0.18;0.18;0.04;0.04], ...
    cfg.weights.output_scale,2.0,1.0);
end

function set = weight_set(name,output,scale,force,force_rate)
set = struct('name',name,'output',output,'output_scale',scale, ...
    'force',force,'force_rate',force_rate);
end

function cfg = apply_weights(cfg,set)
cfg.weights.output = set.output;
cfg.weights.output_scale = set.output_scale;
cfg.weights.force = set.force;
cfg.weights.force_rate = set.force_rate;
end

function roads = define_offline_roads(cfg)
[sb,zb] = extract_carsim_road_table(cfg.road_datasets.class_b);
[sc,zc] = extract_carsim_road_table(cfg.road_datasets.class_c);
s = (0:0.012:120)';
z = raised_cosine_feature(s,30.0,1.0,0.050) + ...
    raised_cosine_feature(s,50.0,0.8,-0.050);
roads.bump_pothole = road_struct('bump_pothole',s,z,7.5);
roads.class_b = road_struct('class_b',sb,zb,8.0);
roads.class_c = road_struct('class_c',sc,zc,8.0);
roads.list = [roads.bump_pothole roads.class_b roads.class_c];
end

function road = road_struct(id,station,elevation,stop_time)
road = struct('id',id,'station',station,'elevation',elevation, ...
    'stop_time',stop_time);
end

function z = raised_cosine_feature(x,start_station,length_m,amplitude_m)
z = zeros(size(x));
mask = x>=start_station & x<=start_station+length_m;
phase = (x(mask)-start_station)/length_m;
z(mask) = 0.5*amplitude_m*(1-cos(2*pi*phase));
end

function metrics = simulate_case(cfg,road,active,stop_time)
data = build_quick_mpc_data(cfg);
dt = cfg.control_dt;
steps = floor(stop_time/dt)+1;
x = zeros(data.sys.nx,1);
u_prev = zeros(data.sys.nu,1);
warm = [];
y_log = zeros(steps,6);
u_log = zeros(steps,2);
iteration_log = zeros(steps,1);
converged_log = true(steps,1);
travel_ratio_log = zeros(steps,1);
speed = 35/3.6;
wheelbase = cfg.vehicle.a+cfg.vehicle.b;

for k = 1:steps
    time = (k-1)*dt;
    W = road_preview(road,speed*time,speed*time-wheelbase,speed,data.N,dt);
    if active
        [u,warm,info] = preview_qp_solve(data,x,u_prev,W,warm);
        iteration_log(k) = info.iterations;
        converged_log(k) = info.converged;
        travel_ratio_log(k) = info.max_predicted_travel_ratio;
    else
        u = zeros(2,1);
    end
    w0 = W(1:4);
    y_log(k,:) = (data.sys.Cy*x+data.sys.Dy*u+data.sys.Fy*w0)';
    u_log(k,:) = u';
    x = data.sys.Ad*x+data.sys.Bd*u+data.sys.Ed*w0;
    u_prev = u;
end

mask = (0:steps-1)'*dt>=0.5;
y = y_log(mask,:);
u = u_log(mask,:);
metrics = struct();
metrics.body_acc_rms = sqrt(mean(y(:,1).^2));
metrics.pitch_acc_rms = sqrt(mean(y(:,2).^2));
metrics.max_front_travel = max(abs(y(:,3)));
metrics.max_rear_travel = max(abs(y(:,4)));
metrics.tire_deflection_rms = sqrt(mean(y(:,5:6).^2,'all'));
metrics.force_rms = sqrt(mean(u.^2,'all'));
metrics.force_peak = max(abs(u),[],'all');
metrics.rate_limit_fraction = mean(any(abs(diff(u_log)/dt)>= ...
    0.999*cfg.force_rate_limit',2));
metrics.force_limit_fraction = mean(any(abs(u)>= ...
    0.999*cfg.force_limit',2));
metrics.qp_converged_fraction = mean(converged_log(mask));
metrics.mean_iterations = mean(iteration_log(mask));
metrics.max_predicted_travel_ratio = max(travel_ratio_log(mask));
end

function W = road_preview(road,front_station,rear_station,speed,N,dt)
t = (0:N)'*dt;
zf = interp_flat(road,front_station+speed*t);
zr = interp_flat(road,rear_station+speed*t);
W = zeros(N*4,1);
for i = 1:N
    W((i-1)*4+(1:4)) = [zf(i);(zf(i+1)-zf(i))/dt; ...
        zr(i);(zr(i+1)-zr(i))/dt];
end
end

function z = interp_flat(road,station)
z = interp1(road.station,road.elevation,station,'linear','extrap');
z(station<=road.station(1)) = road.elevation(1);
z(station>=road.station(end)) = road.elevation(end);
end

function score = normalized_score(active,passive)
score = active.body_acc_rms/max(passive.body_acc_rms,eps) + ...
    0.45*active.pitch_acc_rms/max(passive.pitch_acc_rms,eps) + ...
    0.15*active.max_front_travel/max(passive.max_front_travel,eps) + ...
    0.15*active.max_rear_travel/max(passive.max_rear_travel,eps) + ...
    0.10*active.tire_deflection_rms/max(passive.tire_deflection_rms,eps) + ...
    0.80*active.force_limit_fraction + 0.30*active.rate_limit_fraction;
if active.qp_converged_fraction<0.999
    score = score+10*(1-active.qp_converged_fraction);
end
end

function row = empty_result_row()
row = struct('WeightSet','','Horizon',0,'Road','','Score',0, ...
    'BodyAccRms',0,'BodyAccImprovementPercent',0,'PitchAccRms',0, ...
    'PitchAccImprovementPercent',0,'FrontTravelMax',0, ...
    'RearTravelMax',0,'TireDeflectionRms',0,'ForceRms',0,'ForcePeak',0, ...
    'ForceLimitFraction',0,'RateLimitFraction',0,'QpConvergedFraction',0, ...
    'MeanIterations',0,'MaxPredictedTravelRatio',0);
end

function row = result_row(weight_name,horizon,road_id,active,passive,score)
row = empty_result_row();
row.WeightSet = weight_name;
row.Horizon = horizon;
row.Road = road_id;
row.Score = score;
row.BodyAccRms = active.body_acc_rms;
row.BodyAccImprovementPercent = improvement(passive.body_acc_rms,active.body_acc_rms);
row.PitchAccRms = active.pitch_acc_rms;
row.PitchAccImprovementPercent = improvement(passive.pitch_acc_rms,active.pitch_acc_rms);
row.FrontTravelMax = active.max_front_travel;
row.RearTravelMax = active.max_rear_travel;
row.TireDeflectionRms = active.tire_deflection_rms;
row.ForceRms = active.force_rms;
row.ForcePeak = active.force_peak;
row.ForceLimitFraction = active.force_limit_fraction;
row.RateLimitFraction = active.rate_limit_fraction;
row.QpConvergedFraction = active.qp_converged_fraction;
row.MeanIterations = active.mean_iterations;
row.MaxPredictedTravelRatio = active.max_predicted_travel_ratio;
end

function value = improvement(passive,active)
value = 100*(passive-active)/max(abs(passive),eps);
end

function make_horizon_figure(table_in,path)
fig = figure('Visible','off','Color','w','Position',[100 100 900 520]);
roads = unique(table_in.Road,'stable');
hold on;
for i = 1:numel(roads)
    mask = strcmp(table_in.Road,roads{i});
    plot(table_in.Horizon(mask),table_in.BodyAccImprovementPercent(mask), ...
        '-o','LineWidth',1.4,'DisplayName',strrep(roads{i},'_',' '));
end
grid on; xlabel('Prediction horizon N');
ylabel('Body acceleration RMS improvement (%)');
legend('Location','best');
exportgraphics(fig,path,'Resolution',180);
close(fig);
end
