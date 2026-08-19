function sensor = build_spatial_road_sensor(cfg, station, elevation)
%BUILD_SPATIAL_ROAD_SENSOR Build a distance-indexed road-perception map.
% The map emulates a camera/3-D road sensor output. Sensor errors are fixed
% in road coordinates so repeated controller calls observe the same surface
% estimate instead of drawing unphysical independent time-sample noise.

arguments
    cfg (1,1) struct
    station (:,1) double
    elevation (:,1) double
end

ds = cfg.preview.spatial_resolution_m;
sensor_station = (station(1):ds:station(end))';
if sensor_station(end) < station(end)
    sensor_station(end+1,1) = station(end);
end
truth = interp1(station, elevation, sensor_station, 'linear');

stream = RandStream('mt19937ar', 'Seed', cfg.preview.random_seed);
noise = cfg.preview.sensor_noise_std_m * randn(stream, size(sensor_station));
correlation_samples = max(1, round( ...
    cfg.preview.noise_correlation_length_m / ds));
if correlation_samples > 1
    noise = movmean(noise, correlation_samples, 'Endpoints', 'shrink');
    noise_rms = sqrt(mean(noise.^2));
    if noise_rms > eps
        noise = noise * cfg.preview.sensor_noise_std_m / noise_rms;
    end
end

sensor = struct( ...
    'station', sensor_station, ...
    'elevation', truth + cfg.preview.sensor_bias_m + noise, ...
    'truth', truth, ...
    'spatial_resolution_m', ds, ...
    'range_m', cfg.preview.sensor_range_m);
end
