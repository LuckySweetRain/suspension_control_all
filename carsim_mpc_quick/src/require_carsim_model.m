function require_carsim_model(cfg)
%REQUIRE_CARSIM_MODEL Check the checked-in model; no automatic builder exists.
if ~isfile(cfg.model_file)
    error('CarSimMPC:MissingModel', ...
        'Missing Simulink model: %s. Restore the checked-in .slx model before running.', ...
        cfg.model_file);
end
end
