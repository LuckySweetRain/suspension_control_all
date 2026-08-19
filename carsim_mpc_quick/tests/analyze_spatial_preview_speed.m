function summary = analyze_spatial_preview_speed(speeds_kph)
%ANALYZE_SPATIAL_PREVIEW_SPEED Relate speed, MPC horizon, and sensor range.
if nargin<1, speeds_kph=[10;20;35;50;80;120]; end
speeds_kph=speeds_kph(:);
cfg=quick_mpc_config();
speed_mps=speeds_kph/3.6;
required_distance=speed_mps*cfg.horizon*cfg.control_dt;
used_distance=min(required_distance,cfg.preview.sensor_range_m);
coverage=used_distance./max(required_distance,eps);
sensor_time=cfg.preview.sensor_range_m./max(speed_mps,eps);
covered_steps=min(cfg.horizon,floor( ...
    cfg.preview.sensor_range_m./max(speed_mps*cfg.control_dt,eps)));
steps_to_use_full_range=ceil( ...
    cfg.preview.sensor_range_m./max(speed_mps*cfg.control_dt,eps));

summary=table(speeds_kph,speed_mps,required_distance,used_distance,coverage, ...
    sensor_time,covered_steps,steps_to_use_full_range, ...
    'VariableNames',{'Speed_kph','Speed_mps','MpcRequiredDistance_m', ...
    'SensorUsedDistance_m','CoverageFraction','SensorCoverageTime_s', ...
    'CoveredMpcSteps','StepsToUseFullSensorRange'});

out_dir=fullfile(cfg.results_dir,'spatial_preview');
if ~isfolder(out_dir), mkdir(out_dir); end
writetable(summary,fullfile(out_dir,'speed_distance_mapping.csv'));

fig=figure('Visible','off','Color','w','Position',[100 100 900 520]);
plot(speeds_kph,required_distance,'-o','LineWidth',1.5, ...
    'DisplayName','MPC required distance'); hold on;
yline(cfg.preview.sensor_range_m,'--','LineWidth',1.5, ...
    'DisplayName','Sensor range');
plot(speeds_kph,used_distance,'-s','LineWidth',1.5, ...
    'DisplayName','Usable preview distance');
grid on; xlabel('Vehicle speed (km/h)'); ylabel('Preview distance (m)');
legend('Location','northwest');
exportgraphics(fig,fullfile(out_dir,'speed_distance_mapping.png'),'Resolution',180);
close(fig);
disp(summary);
end
