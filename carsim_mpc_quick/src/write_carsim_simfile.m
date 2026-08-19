function write_carsim_simfile(cfg)
%WRITE_CARSIM_SIMFILE Create a 64-bit MATLAB-specific VS Solver simfile.
if ~isfolder(cfg.generated_dir), mkdir(cfg.generated_dir); end
result_prefix=fullfile(cfg.results_dir,'LastRun');
if ~isfolder(cfg.results_dir), mkdir(cfg.results_dir); end
dll_file='E:\Carsim2020.0_Prog\Programs\solvers\carsim_64.dll';
assert(isfile(dll_file),'64-bit CarSim solver DLL not found: %s',dll_file);

% CarSim advances its equations with the tstep stored in the run parameter
% file. EXT_MODEL_STEP alone does not override that value. Keeping the two
% steps different makes physical distance and Simulink time drift apart.
% Prepare a local run file so the source CarSim database remains unchanged.
run_text=fileread(cfg.carsim_run_par);
tstep_expression='(?m)^tstep\s+[^\r\n]+';
matches=regexp(run_text,tstep_expression,'match');
assert(isscalar(matches),'Expected exactly one tstep entry in %s',cfg.carsim_run_par);
run_text=regexprep(run_text,tstep_expression, ...
    sprintf('tstep %.8f',cfg.carsim_step),'once');

% Keep the vehicle target speed explicit in the local run file. This lets
% speed sweeps reuse one CarSim database without modifying its source PAR.
speed_expression='(?m)^SPEED_TARGET_CONSTANT\s+[^\r\n]+';
speed_matches=regexp(run_text,speed_expression,'match');
assert(isscalar(speed_matches), ...
    'Expected exactly one SPEED_TARGET_CONSTANT entry in %s',cfg.carsim_run_par);
run_text=regexprep(run_text,speed_expression, ...
    sprintf('SPEED_TARGET_CONSTANT %.8f',cfg.target_speed_kph),'once');
runtime_run_par=fullfile(cfg.generated_dir,'runtime_run.par');
run_fid=fopen(runtime_run_par,'w');
assert(run_fid>=0,'Cannot write %s',runtime_run_par);
run_cleanup=onCleanup(@()fclose(run_fid));
run_count=fwrite(run_fid,run_text,'char');
assert(run_count==numel(run_text),'Incomplete write to %s',runtime_run_par);

fid=fopen(cfg.local_simfile,'w'); assert(fid>=0,'Cannot write %s',cfg.local_simfile);
c=onCleanup(@()fclose(fid));
fprintf(fid,'SIMFILE\n\n');
fprintf(fid,'INPUT %s\n',runtime_run_par);
fprintf(fid,'ECHO %s_echo.par\n',result_prefix);
fprintf(fid,'LOGFILE %s_log.txt\n',result_prefix);
fprintf(fid,'FINAL %s_end.par\n',result_prefix);
fprintf(fid,'ERDFILE %s.vs\n',result_prefix);
fprintf(fid,'PROGDIR E:\\Carsim2020.0_Prog\\\n');
fprintf(fid,'DATADIR %s\\\n',cfg.carsim_db_dir);
fprintf(fid,'VEHICLE_CODE i_i\n');
fprintf(fid,'EXT_MODEL_STEP %.8f\n',cfg.carsim_step);
fprintf(fid,'PORTS_IMP 1,8\nPORTS_EXP 1,77\n');
fprintf(fid,'DLLFILE %s\nEND\n',dll_file);
end
