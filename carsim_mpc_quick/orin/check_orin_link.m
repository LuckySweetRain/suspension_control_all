function report = check_orin_link(host)
%CHECK_ORIN_LINK MATLAB Java TCP, duplicate Outputs caching, hash/fallback checks.
if nargin<1, host='192.168.50.2'; end
folder=fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))),'results','orin');
fixture=load(fullfile(folder,'parity_fixture.mat'));
settings=struct('host',host,'port',55001,'timeout_ms',1000, ...
    'sha256',strtrim(fileread(fullfile(folder,'quick_mpc_data.sha256'))));
assignin('base','ORIN_TCP_CONFIG',settings);
orin_tcp_runtime('reset',[],0);
cleanup=onCleanup(@()orin_tcp_runtime('close',[],0)); %#ok<NASGU>
n=size(fixture.exports,2); forces=zeros(8,n); elapsed=zeros(n,1);
for k=1:n
    y=[zeros(38,1);fixture.exports(:,k)];
    start=tic;
    [forces(:,k),dg]=orin_tcp_runtime('step',y,(k-1)*0.01);
    elapsed(k)=toc(start)*1000;
    assert(dg(5)==1,'Unexpected fallback during link test');
    [cached,~]=orin_tcp_runtime('step',y,(k-1)*0.01);
    assert(isequal(cached,forces(:,k)),'Duplicate Outputs was not cached');
end
err=max(abs(forces-fixture.reference),[],'all');
assert(err<1e-5,'Remote controller differs from MATLAB reference');
% Wrong data identity must latch passive damping, even if subsequent input is good.
settings.sha256='intentional-mismatch'; assignin('base','ORIN_TCP_CONFIG',settings);
orin_tcp_runtime('reset',[],0);
y=[zeros(38,1);fixture.exports(:,1)];
[safe,dg]=orin_tcp_runtime('step',y,0);
assert(dg(5)==0 && max(abs(safe-[3*y(61:64);3*y(61:64)]))<1e-8, ...
    'Hash mismatch failed to latch passive damping');
orin_tcp_runtime('close',[],0);
report=struct('samples',n,'max_force_error_N',err,'tcp_step_ms',elapsed, ...
    'median_ms',median(elapsed),'max_ms',max(elapsed), ...
    'duplicate_call_and_hash_fallback_passed',true);
save(fullfile(folder,'matlab_link_check.mat'),'report');
disp(report);
end
