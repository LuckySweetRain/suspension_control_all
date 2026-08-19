function [x,state] = carsim_half_car_state(y,state,cfg)
%CARSIM_HALF_CAR_STATE Reconstruct the reduced half-car state from 39 exports.
% CarSim units: Vz km/h, Pitch/AVy deg(/s), jounce mm(/s).
if isempty(state) || ~isfield(state,'initialized') || ~state.initialized
    state.initialized=true;
    state.z0=y(1); state.pitch0=y(4);
    state.jounce0=[mean(y([7 9]));mean(y([8 10]))];
end
p=cfg.vehicle;
zb=y(1)-state.z0;
dzb=y(2)/3.6;
theta=p.pitch_measurement_polarity*deg2rad(y(4)-state.pitch0);
dtheta=p.pitch_measurement_polarity*deg2rad(y(5));
jf=(mean(y([7 9]))-state.jounce0(1))/1000;
jr=(mean(y([8 10]))-state.jounce0(2))/1000;
jrf=mean(y([11 13]))/1000;
jrr=mean(y([12 14]))/1000;
zwf=zb+p.a*theta+jf;
zwr=zb-p.b*theta+jr;
dzwf=dzb+p.a*dtheta+jrf;
dzwr=dzb-p.b*dtheta+jrr;
x=[zb;dzb;theta;dtheta;zwf;dzwf;zwr;dzwr];
end
