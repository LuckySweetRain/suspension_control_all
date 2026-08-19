function summary = run_carsim_mpc_horizon_sweep(active_scale,horizons,stop_time)
%RUN_CARSIM_MPC_HORIZON_SWEEP Compare preview horizons in CarSim.
if nargin < 2 || isempty(horizons), horizons = [10 15 20 30]; end
if nargin < 3 || isempty(stop_time), stop_time = 12.0; end

root = fileparts(mfilename('fullpath'));
addpath(root,fullfile(root,'src'));
cfg0 = quick_mpc_config();
if nargin < 1 || isempty(active_scale), active_scale = cfg0.active_scale; end
if ~isfile(cfg0.model_file), build_carsim_mpc_model(); end
addpath(cfg0.solver_library_dir);

out_dir = fullfile(cfg0.results_dir,'horizon_sweep');
if ~isfolder(out_dir), mkdir(out_dir); end
rows = repmat(empty_row(),numel(horizons),1);
old_dir = pwd;
cleanup = onCleanup(@() cd(old_dir)); %#ok<NASGU>
cd(cfg0.carsim_db_dir);

for i = 1:numel(horizons)
    cfg = cfg0;
    cfg.horizon = horizons(i);
    cfg.stop_time = stop_time;
    cfg.results_dir = fullfile(out_dir,sprintf('N%d',cfg.horizon));
    if ~isfolder(cfg.results_dir), mkdir(cfg.results_dir); end
    write_carsim_simfile(cfg);
    fprintf('\n=== CarSim horizon sweep: N=%d ===\n',cfg.horizon);
    passive = run_case(cfg,0.0);
    active = run_case(cfg,active_scale);
    save(fullfile(cfg.results_dir,'comparison.mat'), ...
        'passive','active','cfg','active_scale','-v7.3');
    rows(i) = make_row(cfg,passive.metrics,active.metrics);
end

summary = struct2table(rows);
writetable(summary,fullfile(out_dir,'carsim_horizon_sweep.csv'));
make_figure(summary,fullfile(out_dir,'carsim_horizon_sweep.png'));
disp(summary);
end

function out = run_case(cfg,active_scale)
cfg.active_scale = active_scale;
data = build_quick_mpc_data(cfg);
save(cfg.data_file,'data');
half_car_preview_qp_runtime('reset',zeros(39,1));
load_system(cfg.model_file);
sim_out = sim(cfg.model_file,'StopTime',num2str(cfg.stop_time));
close_system('carsim_half_car_qp_mpc',0);
out.exports = sim_out.carsim_exports;
out.forces = sim_out.mpc_corner_forces;
out.diagnostics = sim_out.mpc_diagnostics;
y = canonical_exports(out.exports.signals.values);
forces = out.forces.signals.values;
diagnostics = out.diagnostics.signals.values;
out.metrics = struct( ...
    'body_acc_rms_mps2',sqrt(mean((y(:,3)*9.80665).^2)), ...
    'pitch_acc_rms_radps2',sqrt(mean(deg2rad(y(:,6)).^2)), ...
    'max_front_jounce_m',max(abs(y(:,[7 9])),[],'all')/1000, ...
    'max_rear_jounce_m',max(abs(y(:,[8 10])),[],'all')/1000, ...
    'max_corner_force_N',max(abs(forces),[],'all'), ...
    'max_qp_constraint_violation',max(diagnostics(:,4)), ...
    'qp_converged_fraction',mean(diagnostics(:,5)>0.5));
end

function y = canonical_exports(raw)
if size(raw,2)==77
    y = raw(:,39:77);
elseif size(raw,2)==39
    y = raw;
else
    error('Expected 39 or 77 CarSim exports, received %d.',size(raw,2));
end
end

function row = empty_row()
row = struct('Horizon',0,'PreviewTime_s',0,'BodyAccPassive',0, ...
    'BodyAccMPC',0,'BodyAccImprovementPercent',0,'PitchAccPassive',0, ...
    'PitchAccMPC',0,'PitchAccImprovementPercent',0, ...
    'FrontTravelMPC',0,'RearTravelMPC',0,'MaxCornerForce',0, ...
    'QpConvergedFraction',0,'MaxConstraintViolation',0);
end

function row = make_row(cfg,passive,active)
row = empty_row();
row.Horizon = cfg.horizon;
row.PreviewTime_s = cfg.horizon*cfg.control_dt;
row.BodyAccPassive = passive.body_acc_rms_mps2;
row.BodyAccMPC = active.body_acc_rms_mps2;
row.BodyAccImprovementPercent = improvement( ...
    passive.body_acc_rms_mps2,active.body_acc_rms_mps2);
row.PitchAccPassive = passive.pitch_acc_rms_radps2;
row.PitchAccMPC = active.pitch_acc_rms_radps2;
row.PitchAccImprovementPercent = improvement( ...
    passive.pitch_acc_rms_radps2,active.pitch_acc_rms_radps2);
row.FrontTravelMPC = active.max_front_jounce_m;
row.RearTravelMPC = active.max_rear_jounce_m;
row.MaxCornerForce = active.max_corner_force_N;
row.QpConvergedFraction = active.qp_converged_fraction;
row.MaxConstraintViolation = active.max_qp_constraint_violation;
end

function value = improvement(passive,active)
value = 100*(passive-active)/max(abs(passive),eps);
end

function make_figure(summary,path)
fig = figure('Visible','off','Color','w','Position',[100 100 900 520]);
plot(summary.Horizon,summary.BodyAccImprovementPercent,'-o','LineWidth',1.5, ...
    'DisplayName','Body acceleration RMS');
hold on;
plot(summary.Horizon,summary.PitchAccImprovementPercent,'-s','LineWidth',1.5, ...
    'DisplayName','Pitch acceleration RMS');
grid on; xlabel('Prediction horizon N'); ylabel('Improvement (%)');
legend('Location','best');
exportgraphics(fig,path,'Resolution',180);
close(fig);
end
