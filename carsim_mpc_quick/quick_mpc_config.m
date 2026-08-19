function cfg = quick_mpc_config()
%QUICK_MPC_CONFIG Configuration for the CarSim half-car preview QP-MPC demo.

root = fileparts(mfilename('fullpath'));
cfg.project_root = root;
cfg.carsim_db_dir = 'E:\Desktop\workspace\CarSim_database\carsim_active_suspension';
cfg.carsim_template = fullfile(cfg.carsim_db_dir, 'suspension.slx');
cfg.carsim_run_par = fullfile(cfg.carsim_db_dir, 'Results', ...
    'Run_db4bb818-8634-450e-8fee-ebd655e9283b', 'Run_all.par');
cfg.road_datasets.class_b = fullfile(cfg.carsim_db_dir, 'Roads', 'dZ_Map', ...
    'RdMapZ_b7f8461a-c0a5-450a-aab1-73a3497ceb06.par');
cfg.road_datasets.class_c = fullfile(cfg.carsim_db_dir, 'Roads', 'dZ_Map', ...
    'RdMapZ_a909c49d-e030-436b-9408-dbdd507c8208.par');
cfg.solver_library_dir = 'E:\Carsim2020.0_Prog\Programs\solvers\Matlab84+';

cfg.control_dt = 0.01;
cfg.target_speed_kph = 35.0;
% Ten CarSim integration steps per 10 ms control interval retain numerical
% stability for the current nonlinear suspension model while halving the
% integration count relative to the former 0.5 ms setting. A synchronized
% 2 ms step was tested but made the CarSim vehicle model unstable.
cfg.carsim_step = 0.001;
cfg.horizon = 20;
cfg.stop_time = 20.0;

% Distance-indexed road perception. The sensor supplies a spatial road map;
% the controller maps this map to its time-domain prediction grid using the
% measured vehicle speed. The 3 m range is a provisional calibration value
% and should be replaced by the usable range of the selected camera/3-D sensor.
cfg.preview.mode = 'spatial_sensor';
cfg.preview.sensor_range_m = 3.0;
cfg.preview.spatial_resolution_m = 0.05;
cfg.preview.minimum_speed_mps = 0.5;
cfg.preview.confidence_floor = 0.50;
cfg.preview.far_uncertainty_gain = 2.0;
cfg.preview.sensor_noise_std_m = 0.0;
cfg.preview.sensor_bias_m = 0.0;
cfg.preview.noise_correlation_length_m = 0.20;
cfg.preview.random_seed = 20260818;

% Box-bounded road-perception error used to tighten the predicted travel
% constraints. Height error is in metres and slope error is dz/ds.
cfg.robust.enabled = true;
cfg.robust.height_error_bound_m = 0.003;
cfg.robust.slope_error_bound = 0.010;
cfg.robust.unobserved_height_error_bound_m = 0.030;
cfg.robust.unobserved_slope_error_bound = 0.050;
cfg.robust.max_travel_tightening_ratio = 0.25;

% Deployment model. 'ideal_force' preserves the active-suspension upper
% bound used in the main report. 'persistent_force' adds decision hold and
% force-actuator lag. 'cdc' maps the desired axle force to discrete passive
% damping modes, adds a minimum dwell time, and applies first-order response.
cfg.deployment.mode = 'ideal_force';
cfg.deployment.decision_period_s = 0.05;
cfg.deployment.minimum_dwell_s = 0.10;
cfg.deployment.response_time_constant_s = 0.03;
cfg.deployment.force_quantization_N = 100.0;
cfg.deployment.cdc_damping_levels_N_per_mmps = [1.5; 2.25; 3.0; 4.0; 5.0];
cfg.deployment.cdc_velocity_threshold_mmps = 5.0;

% CarSim D-class sedan reduced to one front and one rear axle.
cfg.vehicle.mb = 1370.0;
cfg.vehicle.Ip = 1972.8;
cfg.vehicle.mwf = 179.8;
cfg.vehicle.mwr = 130.4;
cfg.vehicle.a = 1.110;
cfg.vehicle.b = 1.756;
cfg.vehicle.motion_ratio_f = 0.65144;
cfg.vehicle.motion_ratio_r = 0.79700;
% Sign from a positive CarSim IMP_FD command to the half-car generalized
% suspension force. Keep separate from active_scale so the interface sign
% can be identified without changing the deployed command amplitude.
cfg.vehicle.import_force_polarity = 1.0;
% CarSim AVy/Pitch follow the vehicle Y-axis rotation convention, opposite
% to the half-car model's positive nose-up theta convention.
cfg.vehicle.pitch_measurement_polarity = -1.0;
cfg.vehicle.kf = 2 * 153e3 * cfg.vehicle.motion_ratio_f^2;
cfg.vehicle.kr = 2 * 82e3 * cfg.vehicle.motion_ratio_r^2;
% The imported passive force is 3 N/(mm/s) = 3000 N*s/m at each damper.
% Convert the two corner dampers to an axle-equivalent wheel damping with
% 2*c_corner*gamma^2. This gives 2546.2 and 3811.3 N*s/m.
cfg.vehicle.corner_damper_Ns_per_m = 3000.0;
cfg.vehicle.cf = 2 * cfg.vehicle.corner_damper_Ns_per_m * ...
    cfg.vehicle.motion_ratio_f^2;
cfg.vehicle.cr = 2 * cfg.vehicle.corner_damper_Ns_per_m * ...
    cfg.vehicle.motion_ratio_r^2;
cfg.vehicle.ktf = 400e3;
cfg.vehicle.ktr = 400e3;
cfg.vehicle.ctf = 3000.0;
cfg.vehicle.ctr = 3000.0;

% Two axle-force decisions. Each axle command is divided equally left/right.
% With alpha=1 each axle command is split into two corner commands. A
% 2000 N axle limit leaves about 1000 N/corner active-force headroom above
% the observed 3.9 kN passive peak and avoids downstream 5 kN clipping.
cfg.force_limit = [2000; 2000];
cfg.force_rate_limit = [1.0e5; 1.0e5]; % N/s
% Symmetric dynamic-jounce limits about the initialized equilibrium. The
% working CarSim Class-C/sine-sweep baseline reaches about 67 mm at the
% front axle, so the former 45/55 mm bounds made the online QP infeasible
% before the actuator could recover the state. Use the 80 mm working
% envelope of this CarSim suspension model; the constraint remains hard.
cfg.suspension_limit = [0.080; 0.080];
cfg.carsim_damper_gain = 3.0; % N per (mm/s), matches the working CarSim baseline
cfg.corner_force_limit = 5000.0;
% Deploy the optimized damper command without the former 0.10 residual
% attenuation. The prediction B matrix still includes this scale and the
% suspension motion ratios, so changing it remains model-consistent.
cfg.active_scale = 1.00;

% Dimensionless performance-output weights and scales.
% CarSim-in-the-loop screening favors comfort terms over the former travel-
% dominant weights. The previous normalized travel penalty reduced jounce
% but degraded body and pitch acceleration in the nonlinear vehicle model.
cfg.weights.output = [5.00; 2.00; 0.020; 0.020; 0.010; 0.010];
cfg.weights.output_scale = [1.0; 0.75; 0.040; 0.050; 0.025; 0.025];
cfg.weights.force = 0.40;
cfg.weights.force_rate = 0.20;

% Toolbox-free OSQP-style ADMM settings.
cfg.admm.rho = 1.0;
cfg.admm.sigma = 1.0e-7;
cfg.admm.max_iter = 800;
cfg.admm.abs_tol = 2.0e-4;
cfg.admm.rel_tol = 2.0e-4;

cfg.generated_dir = fullfile(root, 'generated');
cfg.data_file = fullfile(cfg.generated_dir, 'quick_mpc_data.mat');
cfg.local_simfile = fullfile(cfg.generated_dir, 'simfile_mpc.sim');
cfg.model_file = fullfile(root, 'carsim_half_car_qp_mpc.slx');
cfg.results_dir = fullfile(fileparts(root), 'results', 'carsim_mpc_quick');
end
