function sys = half_car_linear_model(cfg)
%HALF_CAR_LINEAR_MODEL Linear 4-DOF half-car model about static equilibrium.
% x = [zb dzb theta dtheta zwf dzwf zwr dzwr]'
% u = [front_axle_damper_command rear_axle_damper_command]'. Each command
% is split equally left/right before deployment. Its generalized axle force
% is alpha*gamma*u, matching the CarSim execution chain.
% w = [front_road front_road_speed rear_road rear_road_speed]'

p = cfg.vehicle;
nx = 8; nu = 2; nw = 4;
input_effectiveness = cfg.active_scale * ...
    p.import_force_polarity * [p.motion_ratio_f; p.motion_ratio_r];

    function dx = dynamics(x, u, w)
        df = x(1) + p.a*x(3) - x(5);
        ddf = x(2) + p.a*x(4) - x(6);
        dr = x(1) - p.b*x(3) - x(7);
        ddr = x(2) - p.b*x(4) - x(8);
        fsf = p.kf*df + p.cf*ddf;
        fsr = p.kr*dr + p.cr*ddr;
        ftf = p.ktf*(x(5)-w(1)) + p.ctf*(x(6)-w(2));
        ftr = p.ktr*(x(7)-w(3)) + p.ctr*(x(8)-w(4));
        u_effective = input_effectiveness .* u;
        dx = zeros(8,1);
        dx(1) = x(2);
        dx(2) = -(fsf+fsr)/p.mb + ...
            (u_effective(1)+u_effective(2))/p.mb;
        dx(3) = x(4);
        dx(4) = (-p.a*fsf+p.b*fsr)/p.Ip + ...
            (p.a*u_effective(1)-p.b*u_effective(2))/p.Ip;
        dx(5) = x(6);
        dx(6) = (fsf-ftf-u_effective(1))/p.mwf;
        dx(7) = x(8);
        dx(8) = (fsr-ftr-u_effective(2))/p.mwr;
    end

A = zeros(nx); B = zeros(nx,nu); E = zeros(nx,nw);
for i = 1:nx
    e = zeros(nx,1); e(i)=1;
    A(:,i) = dynamics(e, zeros(nu,1), zeros(nw,1));
end
for i = 1:nu
    e = zeros(nu,1); e(i)=1;
    B(:,i) = dynamics(zeros(nx,1), e, zeros(nw,1));
end
for i = 1:nw
    e = zeros(nw,1); e(i)=1;
    E(:,i) = dynamics(zeros(nx,1), zeros(nu,1), e);
end

% Exact zero-order-hold discretization without Control System Toolbox.
aug = [A B E; zeros(nu+nw,nx+nu+nw)];
Md = expm(aug*cfg.control_dt);
sys.A = A; sys.B = B; sys.E = E;
sys.Ad = Md(1:nx,1:nx);
sys.Bd = Md(1:nx,nx+(1:nu));
sys.Ed = Md(1:nx,nx+nu+(1:nw));
sys.nx = nx; sys.nu = nu; sys.nw = nw;
sys.input_effectiveness = input_effectiveness;

% y = [body acc; pitch acc; front/rear travel; front/rear tire deflection].
sys.Cy = zeros(6,nx); sys.Dy = zeros(6,nu); sys.Fy = zeros(6,nw);
sys.Cy(1,:) = A(2,:); sys.Dy(1,:) = B(2,:); sys.Fy(1,:) = E(2,:);
sys.Cy(2,:) = A(4,:); sys.Dy(2,:) = B(4,:); sys.Fy(2,:) = E(4,:);
sys.Cy(3,[1 3 5]) = [1 p.a -1];
sys.Cy(4,[1 3 7]) = [1 -p.b -1];
sys.Cy(5,5) = 1; sys.Fy(5,1) = -1;
sys.Cy(6,7) = 1; sys.Fy(6,3) = -1;
end
