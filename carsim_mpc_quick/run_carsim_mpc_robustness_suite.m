function suite = run_carsim_mpc_robustness_suite()
%RUN_CARSIM_MPC_ROBUSTNESS_SUITE Test spatial-preview errors on Class C road.
% Sensor errors are fixed in road coordinates. The CarSim road itself is
% unchanged, so differences come from the preview measurement and robust
% travel-constraint tightening rather than from different road realizations.
root = fileparts(mfilename('fullpath'));
addpath(root, fullfile(root, 'src'));
cfg0 = quick_mpc_config();
cfg0.stop_time = 12.0;
cfg0.deployment.mode = 'ideal_force';
require_carsim_model(cfg0);
addpath(cfg0.solver_library_dir);

suite_dir = fullfile(cfg0.results_dir, 'robustness_suite');
if ~isfolder(suite_dir), mkdir(suite_dir); end
old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir)); %#ok<NASGU>
cd(cfg0.carsim_db_dir);

passive_cfg = cfg0;
passive_cfg.results_dir = fullfile(suite_dir, 'passive');
write_carsim_simfile(passive_cfg);
passive = run_case(passive_cfg, 0.0, 0.5);

scenarios = define_scenarios();
suite = struct('passive', passive, 'scenarios', scenarios);
for i = 1:numel(scenarios)
    scenario = scenarios(i);
    fprintf('\n=== Robustness case %d/%d: %s ===\n', ...
        i, numel(scenarios), scenario.label);
    cfg = apply_scenario(cfg0, scenario);
    cfg.results_dir = fullfile(suite_dir, scenario.id);
    write_carsim_simfile(cfg);
    active = run_case(cfg, cfg.active_scale, 0.5);
    suite.(scenario.id) = struct('scenario', scenario, 'active', active, 'cfg', cfg);
    result = suite.(scenario.id); %#ok<NASGU>
    save(fullfile(cfg.results_dir, 'comparison.mat'), 'result', 'passive', '-v7.3');
end

summary = make_summary_table(suite, scenarios);
writetable(summary, fullfile(suite_dir, 'robustness_metrics.csv'));
make_summary_figure(summary, fullfile(suite_dir, 'robustness_summary.png'));
save(fullfile(suite_dir, 'robustness_summary.mat'), 'suite', 'summary', '-v7.3');
disp(summary);
end

function scenarios = define_scenarios()
scenarios(1) = scenario_struct('nominal', 'Nominal spatial preview', ...
    true, 3.0, 0.0, 0.0);
scenarios(2) = scenario_struct('moderate_robust', ...
    '2 mm noise + 1 mm bias, robust', true, 3.0, 0.002, 0.001);
scenarios(3) = scenario_struct('moderate_no_tightening', ...
    '2 mm noise + 1 mm bias, no tightening', false, 3.0, 0.002, 0.001);
scenarios(4) = scenario_struct('short_range_robust', ...
    '5 mm noise + 2 mm bias, 1.5 m range', true, 1.5, 0.005, 0.002);
end

function scenario = scenario_struct(id, label, robust_enabled, range_m, noise_m, bias_m)
scenario = struct('id', id, 'label', label, ...
    'robust_enabled', robust_enabled, 'sensor_range_m', range_m, ...
    'sensor_noise_std_m', noise_m, 'sensor_bias_m', bias_m);
end

function cfg = apply_scenario(cfg, scenario)
cfg.robust.enabled = scenario.robust_enabled;
cfg.preview.sensor_range_m = scenario.sensor_range_m;
cfg.preview.sensor_noise_std_m = scenario.sensor_noise_std_m;
cfg.preview.sensor_bias_m = scenario.sensor_bias_m;
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
out.metrics = struct( ...
    'body_acc_rms_mps2', sqrt(mean(body_acc.^2)), ...
    'pitch_acc_rms_radps2', sqrt(mean(pitch_acc.^2)), ...
    'max_front_jounce_m', max(abs(jounce(:,[1 3])),[],'all'), ...
    'max_rear_jounce_m', max(abs(jounce(:,[2 4])),[],'all'), ...
    'max_corner_force_N', max(abs(force_eval),[],'all'), ...
    'max_qp_constraint_violation', max(diag_eval(:,4)), ...
    'qp_converged_fraction', mean(diag_eval(:,5) > 0.5), ...
    'mean_preview_distance_m', mean(diag_eval(:,6)), ...
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

function summary = make_summary_table(suite, scenarios)
n = numel(scenarios);
scenario_id = strings(n,1);
scenario_label = strings(n,1);
robust_enabled = false(n,1);
sensor_range_m = zeros(n,1);
noise_std_mm = zeros(n,1);
bias_mm = zeros(n,1);
preview_coverage = zeros(n,1);
body_acc_rms = zeros(n,1);
body_improvement = zeros(n,1);
pitch_acc_rms = zeros(n,1);
pitch_improvement = zeros(n,1);
front_jounce = zeros(n,1);
front_jounce_improvement = zeros(n,1);
max_violation = zeros(n,1);
qp_converged = zeros(n,1);
max_robust_margin = zeros(n,1);
b = suite.passive.metrics;
for i = 1:n
    s = scenarios(i);
    a = suite.(s.id).active.metrics;
    scenario_id(i) = s.id;
    scenario_label(i) = s.label;
    robust_enabled(i) = s.robust_enabled;
    sensor_range_m(i) = s.sensor_range_m;
    noise_std_mm(i) = 1000*s.sensor_noise_std_m;
    bias_mm(i) = 1000*s.sensor_bias_m;
    preview_coverage(i) = a.minimum_preview_coverage;
    body_acc_rms(i) = a.body_acc_rms_mps2;
    body_improvement(i) = improvement(b.body_acc_rms_mps2, body_acc_rms(i));
    pitch_acc_rms(i) = a.pitch_acc_rms_radps2;
    pitch_improvement(i) = improvement(b.pitch_acc_rms_radps2, pitch_acc_rms(i));
    front_jounce(i) = a.max_front_jounce_m;
    front_jounce_improvement(i) = improvement(b.max_front_jounce_m, front_jounce(i));
    max_violation(i) = a.max_qp_constraint_violation;
    qp_converged(i) = a.qp_converged_fraction;
    max_robust_margin(i) = a.max_robust_travel_margin;
end
summary = table(scenario_id, scenario_label, robust_enabled, sensor_range_m, ...
    noise_std_mm, bias_mm, preview_coverage, body_acc_rms, body_improvement, ...
    pitch_acc_rms, pitch_improvement, front_jounce, ...
    front_jounce_improvement, max_violation, qp_converged, ...
    max_robust_margin, ...
    'VariableNames', {'ScenarioId','Scenario','RobustTightening', ...
    'SensorRange_m','NoiseStd_mm','Bias_mm','MinimumPreviewCoverage', ...
    'BodyAccRms_mps2','BodyImprovementPercent','PitchAccRms_radps2', ...
    'PitchImprovementPercent','MaxFrontJounce_m', ...
    'FrontJounceImprovementPercent','MaxQpConstraintViolation', ...
    'QpConvergedFraction','MaxRobustTravelMargin'});
end

function value = improvement(baseline, active)
value = 100*(baseline-active)/max(abs(baseline),eps);
end

function make_summary_figure(summary, output_path)
fig = figure('Visible','off','Color','w','Position',[100 100 1050 560]);
values = [summary.BodyImprovementPercent, summary.PitchImprovementPercent, ...
    summary.FrontJounceImprovementPercent];
bar(values);
grid on;
yline(0,'k-');
ylabel('Improvement relative to passive (%)');
xticks(1:height(summary));
xticklabels(summary.ScenarioId);
xtickangle(15);
legend({'Body acceleration RMS','Pitch acceleration RMS','Front jounce peak'}, ...
    'Location','southoutside','Orientation','horizontal');
exportgraphics(fig, output_path, 'Resolution', 180);
close(fig);
end
