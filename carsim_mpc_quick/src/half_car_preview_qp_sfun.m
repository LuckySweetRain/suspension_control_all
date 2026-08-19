function half_car_preview_qp_sfun(block)
%HALF_CAR_PREVIEW_QP_SFUN Level-2 MATLAB S-function for CarSim co-simulation.
setup(block);
end

function setup(block)
block.NumInputPorts=1; block.NumOutputPorts=2;
block.InputPort(1).Dimensions=77; block.InputPort(1).DirectFeedthrough=true;
block.OutputPort(1).Dimensions=8;
block.OutputPort(2).Dimensions=10;
cfg=quick_mpc_config();
block.SampleTimes=[cfg.control_dt 0];
block.SimStateCompliance='DefaultSimState';
block.RegBlockMethod('Start',@Start);
block.RegBlockMethod('Outputs',@Outputs);
end

function Start(~)
half_car_preview_qp_runtime('reset',zeros(39,1));
end

function Outputs(block)
[u,d]=half_car_preview_qp_runtime('step',block.InputPort(1).Data);
block.OutputPort(1).Data=u;
block.OutputPort(2).Data=d;
end
