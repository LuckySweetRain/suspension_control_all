function [W,meta] = carsim_road_preview(y,data)
%CARSIM_ROAD_PREVIEW Map a spatial sensor profile to the MPC time grid.
% The perception input is indexed by distance. Vehicle speed is used only
% to determine which spatial locations will reach the tyres at k*Ts.

cfg=data.cfg; N=data.N; dt=cfg.control_dt;
xf=mean(y([31 33])); xr=mean(y([32 34]));
v=max(abs(y(39))/3.6,cfg.preview.minimum_speed_mps);

required_distance=(0:N)'*v*dt;
sensor_range=cfg.preview.sensor_range_m;
query_distance=min(required_distance,sensor_range);
front_query=xf+query_distance;
rear_query=xr+query_distance;
zf=flat_interp(data.road_sensor,front_query);
zr=flat_interp(data.road_sensor,rear_query);

distance_ratio=min(query_distance/max(sensor_range,eps),1);
confidence=1-(1-cfg.preview.confidence_floor)*distance_ratio;
unobserved=required_distance>sensor_range;
confidence(unobserved)=0;

height_bound=cfg.robust.height_error_bound_m * ...
    (1+(cfg.preview.far_uncertainty_gain-1)*distance_ratio);
slope_bound=cfg.robust.slope_error_bound * ...
    (1+(cfg.preview.far_uncertainty_gain-1)*distance_ratio);
height_bound(unobserved)=cfg.robust.unobserved_height_error_bound_m;
slope_bound(unobserved)=cfg.robust.unobserved_slope_error_bound;

W=zeros(N*4,1);
W_bound=zeros(N*4,1);
for k=1:N
    ds=query_distance(k+1)-query_distance(k);
    if ds>eps
        dzf_ds=(zf(k+1)-zf(k))/ds;
        dzr_ds=(zr(k+1)-zr(k))/ds;
    else
        dzf_ds=0;
        dzr_ds=0;
    end
    W((k-1)*4+(1:4))=[zf(k);v*dzf_ds;zr(k);v*dzr_ds];
    W_bound((k-1)*4+(1:4))=[height_bound(k);v*slope_bound(k); ...
        height_bound(k);v*slope_bound(k)];
end

meta=struct( ...
    'speed_mps',v, ...
    'required_distance_m',required_distance(end), ...
    'used_distance_m',query_distance(end), ...
    'coverage_fraction',query_distance(end)/max(required_distance(end),eps), ...
    'minimum_confidence',min(confidence), ...
    'W_bound',W_bound);
end

function z=flat_interp(sensor,query)
z=interp1(sensor.station,sensor.elevation,query,'linear','extrap');
z(query<=sensor.station(1))=sensor.elevation(1);
z(query>=sensor.station(end))=sensor.elevation(end);
end
