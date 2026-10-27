function orin_tcp_sfun(block)
%ORIN_TCP_SFUN Replace only the controller in the existing CarSim model.
block.NumInputPorts=1; block.NumOutputPorts=2;
block.InputPort(1).Dimensions=77;
block.InputPort(1).DirectFeedthrough=true;
block.OutputPort(1).Dimensions=8; block.OutputPort(2).Dimensions=10;
block.SampleTimes=[0.01 0];
block.SimStateCompliance='DefaultSimState';
block.RegBlockMethod('Start',@start);
block.RegBlockMethod('Outputs',@outputs);
block.RegBlockMethod('Terminate',@terminate);
end

function start(~)
orin_tcp_runtime('reset',[],0);
end

function outputs(block)
[u,d]=orin_tcp_runtime('step',block.InputPort(1).Data,block.CurrentTime);
block.OutputPort(1).Data=u; block.OutputPort(2).Data=d;
end

function terminate(~)
orin_tcp_runtime('close',[],0);
end
