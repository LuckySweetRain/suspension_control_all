function suite = run_carsim_mpc_road_suite(active_scale,deployment_mode)
%RUN_CARSIM_MPC_ROAD_SUITE Run passive/MPC comparisons on three road cases.
% Cases: a deterministic bump-pothole road and CarSim Class B/Class C roads.
root = fileparts(mfilename('fullpath'));
addpath(root, fullfile(root, 'src'));
cfg0 = quick_mpc_config();
if nargin < 1 || isempty(active_scale), active_scale = cfg0.active_scale; end
if nargin < 2 || isempty(deployment_mode)
    deployment_mode = cfg0.deployment.mode;
end
cfg0.active_scale = active_scale;
cfg0.deployment.mode = char(deployment_mode);
if ~isfile(cfg0.model_file), build_carsim_mpc_model(); end
addpath(cfg0.solver_library_dir);

if strcmpi(cfg0.deployment.mode,'ideal_force')
    suite_name='road_suite';
else
    suite_name=['road_suite_',lower(cfg0.deployment.mode)];
end
suite_dir = fullfile(cfg0.results_dir, suite_name);
run_par_dir = fullfile(cfg0.generated_dir, 'road_runs');
if ~isfolder(suite_dir), mkdir(suite_dir); end
if ~isfolder(run_par_dir), mkdir(run_par_dir); end

roads = define_roads(cfg0);
suite = struct();
old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir)); %#ok<NASGU>
cd(cfg0.carsim_db_dir);

for i = 1:numel(roads)
    road = roads(i);
    fprintf('\n=== Road case %d/%d: %s ===\n', i, numel(roads), road.label);
    cfg = cfg0;
    cfg.stop_time = road.stop_time;
    cfg.results_dir = fullfile(suite_dir, road.id);
    cfg.carsim_run_par = fullfile(run_par_dir, ['Run_all_', road.id, '.par']);
    write_road_run_par(cfg0.carsim_run_par, cfg.carsim_run_par, ...
        road.station, road.elevation, road.label);
    write_carsim_simfile(cfg);

    passive = run_case(cfg, 0.0, 0.5);
    active = run_case(cfg, active_scale, 0.5);
    result = struct('road', road, 'passive', passive, 'active', active, ...
        'active_scale', active_scale, 'deployment_mode', cfg.deployment.mode, ...
        'cfg', cfg);
    suite.(road.id) = result;
    save(fullfile(cfg.results_dir, 'comparison.mat'), 'result', '-v7.3');
end

summary = make_summary_table(suite, roads);
writetable(summary, fullfile(suite_dir, 'road_suite_metrics.csv'));
make_profile_figure(roads, fullfile(suite_dir, 'road_profiles.png'));
make_response_figure(suite, roads, fullfile(suite_dir, 'road_suite_response.png'));
save(fullfile(suite_dir, 'road_suite_summary.mat'), 'suite', 'summary', '-v7.3');
disp(summary);
end

function roads = define_roads(cfg)
[station_b, elevation_b] = extract_carsim_road_table(cfg.road_datasets.class_b);
[station_c, elevation_c] = extract_carsim_road_table(cfg.road_datasets.class_c);

station_bp = (0:0.012:120)';
elevation_bp = zeros(size(station_bp));
elevation_bp = elevation_bp + raised_cosine_feature(station_bp, 30.0, 1.0, 0.050);
elevation_bp = elevation_bp + raised_cosine_feature(station_bp, 50.0, 0.8, -0.050);

roads(1) = road_struct('bump_pothole', 'Bump--pothole, +50/-50 mm', ...
    station_bp, elevation_bp, 12.0, 'Generated raised-cosine features');
roads(2) = road_struct('class_b', 'CarSim Class B #1', ...
    station_b, elevation_b, 12.0, cfg.road_datasets.class_b);
roads(3) = road_struct('class_c', 'CarSim Class C #1', ...
    station_c, elevation_c, 12.0, cfg.road_datasets.class_c);
end

function z = raised_cosine_feature(x, start_station, length_m, amplitude_m)
z = zeros(size(x));
mask = x >= start_station & x <= start_station + length_m;
phase = (x(mask) - start_station) / length_m;
z(mask) = 0.5 * amplitude_m * (1 - cos(2*pi*phase));
end

function road = road_struct(id, label, station, elevation, stop_time, source)
road = struct('id', id, 'label', label, 'station', station, ...
    'elevation', elevation, 'stop_time', stop_time, 'source', source, ...
    'rms_mm', 1000*sqrt(mean((elevation - mean(elevation)).^2)), ...
    'peak_mm', 1000*max(abs(elevation)));
end

function out = run_case(cfg, active_scale, evaluation_start)
cfg.active_scale = active_scale;
data = build_quick_mpc_data(cfg);
save(cfg.data_file, 'data');
half_car_preview_qp_runtime('reset', zeros(39,1));
load_system(cfg.model_file);
sim_out = sim(cfg.model_file, 'StopTime', num2str(cfg.stop_time));
close_system('carsim_half_car_qp_mpc', 0);

out.exports = sim_out.carsim_exports;
out.forces = sim_out.mpc_corner_forces;
out.diagnostics = sim_out.mpc_diagnostics;
y = canonical_exports(out.exports.signals.values);
t = out.exports.time;
mask = t >= evaluation_start;
y_eval = y(mask,:);
force_eval = out.forces.signals.values(out.forces.time >= evaluation_start,:);
diag_eval = out.diagnostics.signals.values(out.diagnostics.time >= evaluation_start,:);

body_acc = y_eval(:,3) * 9.80665;
pitch_acc = deg2rad(y_eval(:,6));
jounce = y_eval(:,7:10) / 1000;
fz = y_eval(:,15:18);
dynamic_fz = fz - mean(fz,1);

out.metrics = struct();
out.metrics.body_acc_rms_mps2 = sqrt(mean(body_acc.^2));
out.metrics.body_acc_peak_mps2 = max(abs(body_acc));
out.metrics.pitch_acc_rms_radps2 = sqrt(mean(pitch_acc.^2));
out.metrics.pitch_acc_peak_radps2 = max(abs(pitch_acc));
out.metrics.max_front_jounce_m = max(abs(jounce(:,[1 3])),[],'all');
out.metrics.max_rear_jounce_m = max(abs(jounce(:,[2 4])),[],'all');
out.metrics.dynamic_tire_load_rms_N = sqrt(mean(dynamic_fz.^2,'all'));
out.metrics.max_corner_force_N = max(abs(force_eval),[],'all');
out.metrics.max_axle_command_N = max(abs(diag_eval(:,1:2)),[],'all');
out.metrics.max_qp_constraint_violation = max(diag_eval(:,4));
out.metrics.qp_converged_fraction = mean(diag_eval(:,5) > 0.5);
out.metrics.mean_qp_iterations = mean(diag_eval(:,3));
if size(diag_eval,2)>=9
    out.metrics.mean_preview_distance_m = mean(diag_eval(:,6));
    out.metrics.minimum_preview_coverage = min(diag_eval(:,9));
    out.metrics.decision_update_fraction = mean(diag_eval(:,8)>0.5);
end
end

function y = canonical_exports(raw)
if size(raw,2) == 77
    y = raw(:,39:77);
elseif size(raw,2) == 39
    y = raw;
else
    error('Expected 39 or 77 CarSim exports, received %d.', size(raw,2));
end
end

function table_out = make_summary_table(suite, roads)
metric_names = { ...
    'body_acc_rms_mps2', 'body_acc_peak_mps2', ...
    'pitch_acc_rms_radps2', 'pitch_acc_peak_radps2', ...
    'max_front_jounce_m', 'max_rear_jounce_m', ...
    'dynamic_tire_load_rms_N', 'max_corner_force_N'};
units = {'m/s^2','m/s^2','rad/s^2','rad/s^2','m','m','N','N'};
rows = cell(numel(roads)*numel(metric_names), 6);
k = 0;
for i = 1:numel(roads)
    result = suite.(roads(i).id);
    for j = 1:numel(metric_names)
        k = k + 1;
        name = metric_names{j};
        passive = result.passive.metrics.(name);
        active = result.active.metrics.(name);
        rows(k,:) = {roads(i).id, roads(i).label, name, passive, active, ...
            100*(passive-active)/max(abs(passive),eps)};
    end
end
table_out = cell2table(rows, 'VariableNames', ...
    {'RoadId','Road','Metric','Passive','PreviewMPC','ImprovementPercent'});
table_out.Unit = repmat(string(units(:)), numel(roads), 1);
end

function make_profile_figure(roads, output_path)
fig = figure('Visible','off','Color','w','Position',[100 100 1100 760]);
layout = tiledlayout(fig, numel(roads), 1, 'TileSpacing','compact', 'Padding','compact');
for i = 1:numel(roads)
    ax = nexttile(layout);
    plot(ax, roads(i).station, 1000*roads(i).elevation, 'k-', 'LineWidth', 1.0);
    grid(ax,'on'); ylabel(ax,'z_r (mm)'); title(ax, roads(i).label, 'Interpreter','none');
    if i == numel(roads), xlabel(ax,'Station (m)'); end
end
exportgraphics(fig, output_path, 'Resolution', 180);
close(fig);
end

function make_response_figure(suite, roads, output_path)
fig = figure('Visible','off','Color','w','Position',[100 100 1250 900]);
layout = tiledlayout(fig, numel(roads), 2, 'TileSpacing','compact', 'Padding','compact');
for i = 1:numel(roads)
    result = suite.(roads(i).id);
    [tp, yp] = response_signals(result.passive);
    [ta, ya] = response_signals(result.active);

    ax1 = nexttile(layout);
    plot(ax1,tp,yp(:,1),'Color',[0.35 0.35 0.35],'LineWidth',0.9); hold(ax1,'on');
    plot(ax1,ta,ya(:,1),'Color',[0.00 0.35 0.75],'LineWidth',0.9);
    grid(ax1,'on'); ylabel(ax1,'a_z (m/s^2)'); title(ax1,roads(i).label,'Interpreter','none');
    if i == 1, legend(ax1,{'Passive','Preview MPC'},'Location','best'); end

    ax2 = nexttile(layout);
    plot(ax2,tp,1000*yp(:,2),'Color',[0.35 0.35 0.35],'LineWidth',0.9); hold(ax2,'on');
    plot(ax2,ta,1000*ya(:,2),'Color',[0.00 0.35 0.75],'LineWidth',0.9);
    grid(ax2,'on'); ylabel(ax2,'Mean |jounce| (mm)');
    if i == numel(roads), xlabel(ax1,'Time (s)'); xlabel(ax2,'Time (s)'); end
end
exportgraphics(fig, output_path, 'Resolution', 180);
close(fig);
end

function [t, signals] = response_signals(run)
y = canonical_exports(run.exports.signals.values);
t = run.exports.time;
signals = [y(:,3)*9.80665, mean(abs(y(:,7:10)),2)/1000];
end
