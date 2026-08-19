function report = run_carsim_mpc_demo(active_scale)
%RUN_CARSIM_MPC_DEMO Compare baseline damping and preview QP-MPC in CarSim.
root=fileparts(mfilename('fullpath'));
addpath(root,fullfile(root,'src'));
cfg=quick_mpc_config();
if nargin<1, active_scale=cfg.active_scale; end
cfg.active_scale=active_scale;
if ~isfile(cfg.model_file), build_carsim_mpc_model(); end
if ~isfolder(cfg.results_dir), mkdir(cfg.results_dir); end
addpath(cfg.solver_library_dir);

old_dir=pwd; cleanup=onCleanup(@()cd(old_dir)); %#ok<NASGU>
write_carsim_simfile(cfg);
cd(cfg.carsim_db_dir);
baseline=run_case(cfg,0.0);
active=run_case(cfg,active_scale);

save(fullfile(cfg.results_dir,'carsim_mpc_run.mat'),'baseline','active','cfg','active_scale');
write_comparison(fullfile(cfg.results_dir,'metrics.csv'),baseline.metrics,active.metrics);
write_report(fullfile(cfg.results_dir,'report.md'),baseline.metrics,active.metrics,cfg,active_scale);
report=struct('baseline',baseline.metrics,'active',active.metrics);
disp(report);
end

function out=run_case(cfg,active_scale)
cfg.active_scale=active_scale;
data=build_quick_mpc_data(cfg);
save(cfg.data_file,'data');
half_car_preview_qp_runtime('reset',zeros(39,1));
load_system(cfg.model_file);
sim_out=sim(cfg.model_file,'StopTime',num2str(cfg.stop_time));
close_system('carsim_half_car_qp_mpc',0);
out.exports=sim_out.carsim_exports;
out.forces=sim_out.mpc_corner_forces;
out.diagnostics=sim_out.mpc_diagnostics;
y=canonical_exports(out.exports.signals.values);
uf=out.forces.signals.values;
dg=out.diagnostics.signals.values;
out.metrics=struct();
out.metrics.body_acc_rms_mps2=rms(y(:,3)*9.80665);
out.metrics.pitch_acc_rms_radps2=rms(deg2rad(y(:,6)));
out.metrics.max_front_jounce_m=max(abs(y(:,[7 9])),[],'all')/1000;
out.metrics.max_rear_jounce_m=max(abs(y(:,[8 10])),[],'all')/1000;
out.metrics.max_corner_force_N=max(abs(uf),[],'all');
out.metrics.max_qp_constraint_violation=max(dg(:,4));
out.metrics.qp_converged_fraction=mean(dg(:,5)>0.5);
end


function y=canonical_exports(raw)
if size(raw,2)==77
    y=raw(:,39:77);
elseif size(raw,2)==39
    y=raw;
else
    error('Expected 39 or 77 CarSim exports, received %d.',size(raw,2));
end
end
function write_comparison(path,b,a)
names=fieldnames(a); n=numel(names);
bv=zeros(n,1); av=zeros(n,1); change=zeros(n,1);
for i=1:n
    bv(i)=b.(names{i}); av(i)=a.(names{i});
    change(i)=metric_improvement(names{i},bv(i),av(i));
end
writetable(table(string(names),bv,av,change, ...
    'VariableNames',{'Metric','Baseline','PreviewMPC','ImprovementPercent'}),path);
end

function write_report(path,b,a,cfg,active_scale)
fid=fopen(path,'w'); assert(fid>=0,'Cannot write %s',path); c=onCleanup(@()fclose(fid)); %#ok<NASGU>
fprintf(fid,'# CarSim Half-Car Preview QP-MPC Result\n\n');
fprintf(fid,'- Prediction horizon: `%d` steps (`%.3f s`)\n',cfg.horizon,cfg.horizon*cfg.control_dt);
fprintf(fid,'- MPC sample time: `%.4f s`\n',cfg.control_dt);
fprintf(fid,'- CarSim integration step: `%.4f s`\n',cfg.carsim_step);
fprintf(fid,'- MPC residual deployment scale: `%.3f`\n\n',active_scale);
fprintf(fid,'| Metric | Baseline | Preview MPC | Improvement |\n|---|---:|---:|---:|\n');
names=fieldnames(a);
for i=1:numel(names)
    imp=metric_improvement(names{i},b.(names{i}),a.(names{i}));
    fprintf(fid,'| %s | %.8g | %.8g | %.3f%% |\n',names{i},b.(names{i}),a.(names{i}),imp);
end
fprintf(fid,'\nBaseline preserves the working CarSim damper law. Preview MPC adds equally split front/rear axle forces. Roll is not controlled.\n');
end
function value=metric_improvement(name,baseline,active)
if abs(baseline)<=eps
    value=NaN;
elseif contains(name,'converged_fraction')
    value=100*(active-baseline)/abs(baseline);
else
    value=100*(baseline-active)/abs(baseline);
end
end
