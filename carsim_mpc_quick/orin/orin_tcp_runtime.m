function [forces, diagnostics] = orin_tcp_runtime(command, exports, time)
%ORIN_TCP_RUNTIME Cached synchronous TCP exchange; a fault latches passive damping.
% This is simulation-time lockstep, not a hard-real-time scheduler.
persistent sock reader writer seq lastTime lastForces lastDiag failed settings
if strcmp(command, 'close')
    if ~isempty(sock), try, sock.close(); catch, end, end
    sock=[]; forces=[]; diagnostics=[]; return
end
if strcmp(command, 'reset')
    if ~isempty(sock), try, sock.close(); catch, end, end
    sock=[]; reader=[]; writer=[]; seq=0; lastTime=[];
    lastForces=[]; lastDiag=[]; failed=false;
    settings=evalin('base','ORIN_TCP_CONFIG');
    assignin('base','ORIN_TCP_FAULT',struct('failed',false,'time',NaN,'reason',''));
    forces=zeros(8,1); diagnostics=zeros(10,1); return
end
if ~isempty(lastTime) && abs(time-lastTime)<1e-10
    forces=lastForces; diagnostics=lastDiag; return
end
y=double(exports(:));
if numel(y)==77, y=y(39:77); end
assert(numel(y)==39 && all(isfinite(y)), 'Invalid CarSim exports');
baseline=min(max(3*y(23:26),-5000),5000);
forces=[baseline;baseline]; diagnostics=[0;0;0;1;0;0;0;0;0;0];
if ~failed
    try
        if isempty(sock)
            sock=javaObject('java.net.Socket');
            endpoint=javaObject('java.net.InetSocketAddress',settings.host,settings.port);
            sock.connect(endpoint,int32(settings.timeout_ms));
            sock.setSoTimeout(int32(settings.timeout_ms)); sock.setTcpNoDelay(true);
            reader=javaObject('java.io.BufferedReader', ...
                javaObject('java.io.InputStreamReader',sock.getInputStream(),'UTF-8'));
            writer=javaObject('java.io.OutputStreamWriter',sock.getOutputStream(),'UTF-8');
        end
        request=struct('version',1,'seq',seq,'time',time,'exports',double(exports(:))');
        writer.write([jsonencode(request), char(10)]); writer.flush();
        line=reader.readLine();
        assert(~isempty(line),'Orin disconnected');
        response=jsondecode(char(line));
        assert(response.version==1 && response.seq==seq,'Response sequence mismatch');
        assert(strcmp(response.sha256,settings.sha256),'MPC data hash mismatch');
        candidate=double(response.forces(:)); dg=double(response.diagnostics(:));
        assert(numel(candidate)==8 && numel(dg)==10 && ...
            all(isfinite([candidate;dg])) && all(abs(candidate)<=5000+1e-8), ...
            'Invalid Orin control response');
        assert(~response.fallback,'Orin solver entered passive fallback');
        forces=candidate; diagnostics=dg; seq=seq+1;
    catch errorInfo
        failed=true;
        assignin('base','ORIN_TCP_FAULT',struct('failed',true,'time',time, ...
            'reason',errorInfo.message));
        if ~isempty(sock), try, sock.close(); catch, end, end
        warning('Orin:PassiveFallback','Passive damping latched at t=%.6f: %s', ...
            time,errorInfo.message);
    end
end
lastTime=time; lastForces=forces; lastDiag=diagnostics;
end
