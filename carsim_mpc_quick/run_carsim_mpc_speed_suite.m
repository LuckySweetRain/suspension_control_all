function suite = run_carsim_mpc_speed_suite(speeds_kph, deployment_mode)
%RUN_CARSIM_MPC_SPEED_SUITE Compare passive/MPC at multiple vehicle speeds.
% Each case is evaluated over the same front-wheel station interval so a
% higher speed does not receive an artificial flat-road tail.
root = fileparts(mfilename('fullpath'));
addpath(root, fullfile(root, 'src'));
cfg0 = quick_mpc_config();
if nargin < 1 || isempty(speeds_kph), speeds_kph = [35 60 80]; end
if nargin < 2 || isempty(deployment_mode), deployment_mode = 'ideal_force'; end
speeds_kph = double(speeds_kph(:));
assert(all(isfinite(speeds_kph) & speeds_kph > 0), ...
    'Vehicle speeds must be finite and positive.');
cfg0.deployment.mode = char(deployment_mode);
if ~isfile(cfg0.model_file), build_carsim_mpc_model(); end
addpath(cfg0.solver_library_dir);

if strcmpi(cfg0.deployment.mode, 'ideal_force')
    suite_name = 'speed_suite';
else
    suite_name = ['speed_suite_', lower(cfg0.deployment.mode)];
end
suite_dir = fullfile(cfg0.results_dir, suite_name);
run_par_dir = fullfile(cfg0.generated_dir, 'speed_runs');
if ~isfolder(suite_dir), mkdir(suite_dir); end
if ~isfolder(run_par_dir), mkdir(run_par_dir); end

roads = define_roads(cfg0);
road_pars = prepare_road_files(cfg0, roads, run_par_dir);
suite = struct('speeds_kph', speeds_kph, 'deployment_mode', cfg0.deployment.mode);
old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir));
cd(cfg0.carsim_db_dir);

for i = 1:numel(speeds_kph)
    speed_kph = speeds_kph(i);
    speed_id = sprintf('speed_%03d', round(speed_kph));
    suite.(speed_id) = struct();
    for j = 1:numel(roads)
        road = roads(j);
        fprintf('\n=== Speed %.1f km/h, road %s ===\n', speed_kph, road.label);
        cfg = cfg0;
        cfg.target_speed_kph = speed_kph;
        cfg.carsim_run_par = road_pars{j};
        cfg.stop_time = road.evaluation_end_m/(speed_kph/3.6) + 0.8;
        cfg.results_dir = fullfile(suite_dir, speed_id, road.id);
        write_carsim_simfile(cfg);

        passive = run_case(cfg, 0.0, road.evaluation_start_m, road.evaluation_end_m);
        active = run_case(cfg, cfg.active_scale, ...
            road.evaluation_start_m, road.evaluation_end_m);
        result = struct('road', road, 'speed_kph', speed_kph, ...
            'passive', passive, 'active', active, 'cfg', cfg);
        save(fullfile(cfg.results_dir, 'comparison.mat'), 'result', '-v7.3');
        suite.(speed_id).(road.id) = struct( ...
            'road', road, 'speed_kph', speed_kph, ...
            'passive', passive.metrics, 'active', active.metrics);
    end
end

summary = make_summary_table(suite, speeds_kph, roads);
suite.summary = summary;
writetable(summary, fullfile(suite_dir, 'speed_suite_metrics.csv'));
make_summary_figure(summary, roads, ...
    fullfile(suite_dir, 'speed_suite_summary.png'));
save(fullfile(suite_dir, 'speed_suite_summary.mat'), ...
    'suite', 'summary', 'roads', '-v7.3');
disp(summary);
end

function roads = define_roads(cfg)
[station_b, elevation_b] = extract_carsim_road_table(cfg.road_datasets.class_b);
[station_c, elevation_c] = extract_carsim_road_table(cfg.road_datasets.class_c);
station_bp = (0:0.012:120)';
elevation_bp = raised_cosine_feature(station_bp, 30.0, 1.0, 0.050) + ...
    raised_cosine_feature(station_bp, 50.0, 0.8, -0.050);

roads(1) = road_struct('bump_pothole', 'Bump--pothole, +50/-50 mm', ...
    station_bp, elevation_bp, 'Generated raised-cosine features');
roads(2) = road_struct('class_b', 'CarSim Class B #1', ...
    station_b, elevation_b, cfg.road_datasets.class_b);
roads(3) = road_struct('class_c', 'CarSim Class C #1', ...
    station_c, elevation_c, cfg.road_datasets.class_c);
end

function road = road_struct(id, label, station, elevation, source)
road = struct('id', id, 'label', label, 'station', station, ...
    'elevation', elevation, 'source', source, ...
    'evaluation_start_m', 10.0, 'evaluation_end_m', 110.0);
end

function z = raised_cosine_feature(x, start_station, length_m, amplitude_m)
z = zeros(size(x));
mask = x >= start_station & x <= start_station + length_m;
phase = (x(mask) - start_station) / length_m;
z(mask) = 0.5 * amplitude_m * (1 - cos(2*pi*phase));
end

function road_pars = prepare_road_files(cfg, roads, run_par_dir)
road_pars = cell(numel(roads),1);
for i = 1:numel(roads)
    road_pars{i} = fullfile(run_par_dir, ['Run_all_', roads(i).id, '.par']);
    write_road_run_par(cfg.carsim_run_par, road_pars{i}, ...
        roads(i).station, roads(i).elevation, roads(i).label);
end
end

function out = run_case(cfg, active_scale, station_start, station_end)
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
front_station = mean(y(:,[31 33]),2);
mask = front_station >= station_start & front_station <= station_end;
assert(nnz(mask) >= 10, ...
    'Vehicle did not traverse the %.1f--%.1f m evaluation interval.', ...
    station_start, station_end);
t_start = t(find(mask,1,'first'));
t_end = t(find(mask,1,'last'));
y_eval = y(mask,:);
force_mask = out.forces.time >= t_start & out.forces.time <= t_end;
diag_mask = out.diagnostics.time >= t_start & out.diagnostics.time <= t_end;
force_eval = out.forces.signals.values(force_mask,:);
diag_eval = out.diagnostics.signals.values(diag_mask,:);

body_acc = y_eval(:,3) * 9.80665;
pitch_acc = deg2rad(y_eval(:,6));
jounce = y_eval(:,7:10) / 1000;
fz = y_eval(:,15:18);
dynamic_fz = fz - mean(fz,1);
out.metrics = struct( ...
    'mean_speed_kph', mean(y_eval(:,39)), ...
    'evaluation_time_s', t_end-t_start, ...
    'body_acc_rms_mps2', sqrt(mean(body_acc.^2)), ...
    'body_acc_peak_mps2', max(abs(body_acc)), ...
    'pitch_acc_rms_radps2', sqrt(mean(pitch_acc.^2)), ...
    'pitch_acc_peak_radps2', max(abs(pitch_acc)), ...
    'max_front_jounce_m', max(abs(jounce(:,[1 3])),[],'all'), ...
    'max_rear_jounce_m', max(abs(jounce(:,[2 4])),[],'all'), ...
    'dynamic_tire_load_rms_N', sqrt(mean(dynamic_fz.^2,'all')), ...
    'max_corner_force_N', max(abs(force_eval),[],'all'), ...
    'max_axle_command_N', max(abs(diag_eval(:,1:2)),[],'all'), ...
    'max_qp_constraint_violation', max(diag_eval(:,4)), ...
    'qp_converged_fraction', mean(diag_eval(:,5) > 0.5), ...
    'mean_preview_distance_m', mean(diag_eval(:,6)), ...
    'mean_required_preview_distance_m', mean(diag_eval(:,7)), ...
    'minimum_preview_coverage', min(diag_eval(:,9)), ...
    'max_robust_travel_margin', max(diag_eval(:,10)));
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

function summary = make_summary_table(suite, speeds_kph, roads)
n = numel(speeds_kph)*numel(roads);
speed = zeros(n,1);
actual_speed = zeros(n,1);
road_id = strings(n,1);
road_label = strings(n,1);
body_passive = zeros(n,1);
body_mpc = zeros(n,1);
body_improvement = zeros(n,1);
pitch_passive = zeros(n,1);
pitch_mpc = zeros(n,1);
pitch_improvement = zeros(n,1);
front_improvement = zeros(n,1);
rear_improvement = zeros(n,1);
tire_load_improvement = zeros(n,1);
preview_required = zeros(n,1);
preview_used = zeros(n,1);
preview_coverage = zeros(n,1);
qp_converged = zeros(n,1);
max_violation = zeros(n,1);
k = 0;
for i = 1:numel(speeds_kph)
    speed_id = sprintf('speed_%03d', round(speeds_kph(i)));
    for j = 1:numel(roads)
        k = k+1;
        item = suite.(speed_id).(roads(j).id);
        p = item.passive;
        a = item.active;
        speed(k) = speeds_kph(i);
        actual_speed(k) = a.mean_speed_kph;
        road_id(k) = roads(j).id;
        road_label(k) = roads(j).label;
        body_passive(k) = p.body_acc_rms_mps2;
        body_mpc(k) = a.body_acc_rms_mps2;
        body_improvement(k) = improvement(body_passive(k), body_mpc(k));
        pitch_passive(k) = p.pitch_acc_rms_radps2;
        pitch_mpc(k) = a.pitch_acc_rms_radps2;
        pitch_improvement(k) = improvement(pitch_passive(k), pitch_mpc(k));
        front_improvement(k) = improvement( ...
            p.max_front_jounce_m, a.max_front_jounce_m);
        rear_improvement(k) = improvement( ...
            p.max_rear_jounce_m, a.max_rear_jounce_m);
        tire_load_improvement(k) = improvement( ...
            p.dynamic_tire_load_rms_N, a.dynamic_tire_load_rms_N);
        preview_required(k) = a.mean_required_preview_distance_m;
        preview_used(k) = a.mean_preview_distance_m;
        preview_coverage(k) = a.minimum_preview_coverage;
        qp_converged(k) = a.qp_converged_fraction;
        max_violation(k) = a.max_qp_constraint_violation;
    end
end
summary = table(speed, actual_speed, road_id, road_label, ...
    body_passive, body_mpc, body_improvement, ...
    pitch_passive, pitch_mpc, pitch_improvement, ...
    front_improvement, rear_improvement, tire_load_improvement, ...
    preview_required, preview_used, preview_coverage, ...
    qp_converged, max_violation, ...
    'VariableNames', {'TargetSpeed_kph','MeanSpeed_kph','RoadId','Road', ...
    'BodyAccPassive_mps2','BodyAccMPC_mps2','BodyImprovementPercent', ...
    'PitchAccPassive_radps2','PitchAccMPC_radps2','PitchImprovementPercent', ...
    'FrontJounceImprovementPercent','RearJounceImprovementPercent', ...
    'TireLoadImprovementPercent','RequiredPreviewDistance_m', ...
    'UsedPreviewDistance_m','MinimumPreviewCoverage', ...
    'QpConvergedFraction','MaxQpConstraintViolation'});
end

function value = improvement(baseline, active)
value = 100*(baseline-active)/max(abs(baseline),eps);
end

function make_summary_figure(summary, roads, output_path)
fig = figure('Visible','off','Color','w','Position',[100 100 1050 820]);
layout = tiledlayout(fig,3,1,'TileSpacing','compact','Padding','compact');
colors = lines(numel(roads));
legend_names = {'Bump--pothole','Class B','Class C'};
metrics = {'BodyImprovementPercent','PitchImprovementPercent'};
ylabels = {'Body acceleration RMS improvement (%)', ...
    'Pitch acceleration RMS improvement (%)'};
for panel = 1:2
    ax = nexttile(layout); hold(ax,'on');
    for j = 1:numel(roads)
        mask = summary.RoadId == roads(j).id;
        plot(ax,summary.TargetSpeed_kph(mask),summary.(metrics{panel})(mask), ...
            '-o','LineWidth',1.5,'Color',colors(j,:), ...
            'DisplayName',legend_names{j});
    end
    yline(ax,0,'k-','HandleVisibility','off'); grid(ax,'on');
    ylabel(ax,ylabels{panel});
    if panel==1, legend(ax,'Location','best'); end
end
ax = nexttile(layout); hold(ax,'on');
mask = summary.RoadId == roads(1).id;
plot(ax,summary.TargetSpeed_kph(mask),100*summary.MinimumPreviewCoverage(mask), ...
    '-s','LineWidth',1.5,'Color',[0.20 0.45 0.70]);
grid(ax,'on'); xlabel(ax,'Target speed (km/h)');
ylabel(ax,'Minimum preview coverage (%)'); ylim(ax,[0 105]);
exportgraphics(fig,output_path,'Resolution',180);
close(fig);
end
