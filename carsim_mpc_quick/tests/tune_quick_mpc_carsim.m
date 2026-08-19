function summary = tune_quick_mpc_carsim()
%TUNE_QUICK_MPC_CARSIM Identify interface polarity and tune MPC weights.
% The sweep is deliberately small: it uses the actual CarSim plant and the
% current Class-C/sine-sweep run for screening, not the internal half-car
% prediction model as a surrogate for closed-loop performance.

root = fileparts(fileparts(mfilename('fullpath')));
addpath(root, fullfile(root, 'src'));
cfg0 = quick_mpc_config();
cfg0.stop_time = 12.0;
if ~isfile(cfg0.model_file), build_carsim_mpc_model(); end
addpath(cfg0.solver_library_dir);
if ~isfolder(cfg0.results_dir), mkdir(cfg0.results_dir); end

old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir)); %#ok<NASGU>
write_carsim_simfile(cfg0);
cd(cfg0.carsim_db_dir);

passive_cfg = cfg0;
passive_cfg.active_scale = 0.0;
passive = run_case(passive_cfg, 0.5);

candidates = define_candidates(cfg0);
polarities = [1.0, -1.0];
rows = repmat(empty_row(), numel(candidates)*numel(polarities), 1);
k = 0;
for ip = 1:numel(polarities)
    for ic = 1:numel(candidates)
        k = k + 1;
        cfg = apply_candidate(cfg0, candidates(ic), polarities(ip));
        fprintf('\n=== CarSim tuning %d/%d: %s, polarity %+.0f ===\n', ...
            k, numel(rows), candidates(ic).name, polarities(ip));
        active = run_case(cfg, 0.5);
        rows(k) = make_row(candidates(ic).name, polarities(ip), passive.metrics, ...
            active.metrics, candidates(ic));
    end
end

summary = struct2table(rows);
summary = sortrows(summary, 'Score', 'ascend');
output_dir = fullfile(cfg0.results_dir, 'carsim_tuning');
if ~isfolder(output_dir), mkdir(output_dir); end
writetable(summary, fullfile(output_dir, 'weight_polarity_sweep.csv'));
save(fullfile(output_dir, 'weight_polarity_sweep.mat'), ...
    'summary', 'passive', 'candidates', 'cfg0');
disp(summary(:, {'Name','Polarity','BodyImprovementPercent', ...
    'PitchImprovementPercent','FrontTravelImprovementPercent', ...
    'RearTravelImprovementPercent','ConvergedFraction', ...
    'MaxConstraintViolation','Score'}));
end

function candidates = define_candidates(cfg)
candidates(1) = candidate('current', cfg.weights.output, 2.0, 1.0);
candidates(2) = candidate('balanced', ...
    [2.5;1.2;0.040;0.040;0.020;0.020], 0.60, 0.30);
candidates(3) = candidate('comfort', ...
    [4.0;1.0;0.015;0.015;0.010;0.010], 0.30, 0.15);
candidates(4) = candidate('body_pitch', ...
    [5.0;2.0;0.020;0.020;0.010;0.010], 0.40, 0.20);
candidates(5) = candidate('moderate', ...
    [3.0;1.5;0.030;0.030;0.010;0.010], 1.00, 0.50);
end

function c = candidate(name, output, force, force_rate)
c = struct('name', name, 'output', output, ...
    'force', force, 'force_rate', force_rate);
end

function cfg = apply_candidate(cfg, c, polarity)
cfg.weights.output = c.output;
cfg.weights.force = c.force;
cfg.weights.force_rate = c.force_rate;
cfg.vehicle.import_force_polarity = polarity;
cfg.active_scale = 1.0;
end

function out = run_case(cfg, evaluation_start)
data = build_quick_mpc_data(cfg);
save(cfg.data_file, 'data');
half_car_preview_qp_runtime('reset', zeros(39,1));
load_system(cfg.model_file);
sim_out = sim(cfg.model_file, 'StopTime', num2str(cfg.stop_time));
close_system('carsim_half_car_qp_mpc', 0);

raw = sim_out.carsim_exports.signals.values;
if size(raw,2) == 77, y = raw(:,39:77); else, y = raw; end
t = sim_out.carsim_exports.time;
mask = t >= evaluation_start;
y = y(mask,:);
force = sim_out.mpc_corner_forces.signals.values;
force = force(sim_out.mpc_corner_forces.time >= evaluation_start,:);
diag = sim_out.mpc_diagnostics.signals.values;
diag = diag(sim_out.mpc_diagnostics.time >= evaluation_start,:);

body_acc = y(:,3)*9.80665;
pitch_acc = deg2rad(y(:,6));
jounce = y(:,7:10)/1000;
dynamic_fz = y(:,15:18) - mean(y(:,15:18),1);
out.metrics = struct( ...
    'body_acc_rms', rms(body_acc), ...
    'pitch_acc_rms', rms(pitch_acc), ...
    'front_travel_max', max(abs(jounce(:,[1 3])),[],'all'), ...
    'rear_travel_max', max(abs(jounce(:,[2 4])),[],'all'), ...
    'tire_load_rms', rms(dynamic_fz,'all'), ...
    'corner_force_max', max(abs(force),[],'all'), ...
    'axis_command_max', max(abs(diag(:,1:2)),[],'all'), ...
    'constraint_violation_max', max(diag(:,4)), ...
    'converged_fraction', mean(diag(:,5) > 0.5));
end

function row = empty_row()
row = struct('Name',"",'Polarity',0,'ForceWeight',0,'RateWeight',0, ...
    'BodyImprovementPercent',0,'PitchImprovementPercent',0, ...
    'FrontTravelImprovementPercent',0,'RearTravelImprovementPercent',0, ...
    'TireLoadImprovementPercent',0,'CornerForceMax',0,'AxisCommandMax',0, ...
    'ConvergedFraction',0,'MaxConstraintViolation',0,'Score',0);
end

function row = make_row(name, polarity, passive, active, candidate_data)
row = empty_row();
row.Name = string(name);
row.Polarity = polarity;
row.ForceWeight = candidate_data.force;
row.RateWeight = candidate_data.force_rate;
row.BodyImprovementPercent = improvement(passive.body_acc_rms, active.body_acc_rms);
row.PitchImprovementPercent = improvement(passive.pitch_acc_rms, active.pitch_acc_rms);
row.FrontTravelImprovementPercent = improvement( ...
    passive.front_travel_max, active.front_travel_max);
row.RearTravelImprovementPercent = improvement( ...
    passive.rear_travel_max, active.rear_travel_max);
row.TireLoadImprovementPercent = improvement(passive.tire_load_rms, active.tire_load_rms);
row.CornerForceMax = active.corner_force_max;
row.AxisCommandMax = active.axis_command_max;
row.ConvergedFraction = active.converged_fraction;
row.MaxConstraintViolation = active.constraint_violation_max;
comfort_ratio = 0.55*active.body_acc_rms/passive.body_acc_rms + ...
    0.30*active.pitch_acc_rms/passive.pitch_acc_rms;
secondary_ratio = 0.05*active.front_travel_max/passive.front_travel_max + ...
    0.05*active.rear_travel_max/passive.rear_travel_max + ...
    0.05*active.tire_load_rms/passive.tire_load_rms;
solver_penalty = 2.0*max(0,0.995-active.converged_fraction) + ...
    2.0*active.constraint_violation_max;
row.Score = comfort_ratio + secondary_ratio + solver_penalty;
end

function value = improvement(passive, active)
value = 100*(passive-active)/max(abs(passive),eps);
end
