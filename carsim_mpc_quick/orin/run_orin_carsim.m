function result = run_orin_carsim(stopTime, host, timeoutMs)
%RUN_ORIN_CARSIM Windows CarSim plant + Orin preview MPC over direct Ethernet.
% Start the service on Orin and deploy results/orin/quick_mpc_data.mat first.
if nargin<1, stopTime=2; end
if nargin<2, host='192.168.50.2'; end
if nargin<3, timeoutMs=100; end
root=fileparts(fileparts(mfilename('fullpath')));
addpath(root,fullfile(root,'src'),fullfile(root,'orin'));
cfg=quick_mpc_config();
folder=fullfile(fileparts(root),'results','orin');
if ~isfolder(folder), mkdir(folder); end
dataPath=fullfile(folder,'quick_mpc_data.mat');
assert(isfile(dataPath),'Run prepare_orin_data first, then deploy the MAT file to Orin.');
hashPath=fullfile(folder,'quick_mpc_data.sha256');
assert(isfile(hashPath),'Missing data hash: generate it during deployment.');
settings=struct('host',host,'port',55001,'timeout_ms',timeoutMs, ...
    'sha256',strtrim(fileread(hashPath)));
assignin('base','ORIN_TCP_CONFIG',settings);
cfg.generated_dir=folder; cfg.results_dir=folder;
cfg.local_simfile=fullfile(folder,'simfile_mpc.sim');
cfg.carsim_step=0.001; cfg.control_dt=0.01;
write_carsim_simfile(cfg);
addpath(cfg.solver_library_dir);
old=pwd; cleanup=onCleanup(@()cd(old)); %#ok<NASGU>
cd(cfg.carsim_db_dir);
source='carsim_half_car_qp_mpc'; target='carsim_orin_tcp';
assert(~bdIsLoaded(source),'Close the original model before building its Orin copy.');
load_system(cfg.model_file);
save_system(source,fullfile(folder,[target,'.slx']));
set_param([target,'/Half-car preview QP-MPC'],'FunctionName','orin_tcp_sfun');
set_param([target,'/CarSim S-Function'],'SIMFILE',cfg.local_simfile);
set_param(target,'StopTime',num2str(stopTime));
save_system(target);
result=sim(target,'StopTime',num2str(stopTime));
fault=evalin('base','ORIN_TCP_FAULT');
exports=result.carsim_exports.signals.values;
times=result.carsim_exports.time;
save(fullfile(folder,'carsim_orin_run.mat'),'result','settings','fault','exports','times');
dg=result.mpc_diagnostics.signals.values;
fprintf('QP converged fraction: %.4f; max normalized QP violation: %.6g\n', ...
    mean(dg(:,5)>0.5),max(dg(:,4)));
assert(~fault.failed,'Orin run entered passive fallback at %.6f: %s',fault.time,fault.reason);
end
