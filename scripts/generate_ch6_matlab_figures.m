function generate_ch6_matlab_figures()
%GENERATE_CH6_MATLAB_FIGURES Regenerate Chapter 6 figures in the IJRNC style.

thisFile = mfilename('fullpath');
projectRoot = fileparts(fileparts(thisFile));
reportRoot = 'E:\Desktop\workspace\active-suspension-midterm-report';
figDir = fullfile(reportRoot, 'figures');
resultDir = fullfile(projectRoot, 'results');
evalDir = fullfile(resultDir, 'report_final_current_20260624');

if ~exist(figDir, 'dir')
    mkdir(figDir);
end

set(groot, 'defaultAxesFontName', 'Times New Roman');
set(groot, 'defaultTextFontName', 'Times New Roman');
set(groot, 'defaultLegendFontName', 'Times New Roman');

metrics = readtable(fullfile(evalDir, 'metrics.csv'), ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');

plotMetricBar(metrics, figDir);
plotClassBPSD(evalDir, figDir);
plotClassBAcceleration(evalDir, figDir);
plotClassBMotion(evalDir, figDir);
plotComfortEnergy(metrics, figDir);
plotGateBehavior(metrics, resultDir, evalDir, figDir);
plotTrainingCurves(resultDir, evalDir, figDir);
plotConvergence(resultDir, evalDir, figDir);
plotBumpResponse(evalDir, figDir);

fprintf('Chapter 6 MATLAB figures saved to: %s\n', figDir);
end

function plotMetricBar(metrics, figDir)
controllers = ["Passive","PID","SPDF","PPO","SI-RPPO"];
labels = ["Passive","PID","SPDF","PPO","SI-RPPO"];
scenarios = ["class_b","class_c","mixed"];
scenarioLabels = ["Class B","Class C","Mixed"];

Y = nan(numel(scenarios), numel(controllers));
for i = 1:numel(scenarios)
    for j = 1:numel(controllers)
        row = find(metrics.Scenario == scenarios(i) & normalizeController(metrics.Controller) == controllers(j), 1);
        if ~isempty(row)
            Y(i, j) = metrics.BodyAccRMS_mps2(row);
        end
    end
end

fig = newFig('singleTall');
ax = axes(fig);
b = bar(ax, Y, 'grouped', 'BarWidth', 0.82);
for j = 1:numel(b)
    st = ctrlStyle(controllers(j));
    b(j).FaceColor = st.Color;
    b(j).EdgeColor = 'k';
    b(j).LineWidth = 0.35;
end
set(ax, 'XTickLabel', scenarioLabels);
ylabel(ax, '$a_{b,\mathrm{rms}}$ (m/s$^2$)', 'Interpreter', 'latex');
legend(ax, labels, 'Location', 'northwest', 'NumColumns', 3);
styleAxis(ax);
ylim(ax, [0, max(Y(:), [], 'omitnan') * 1.18]);
saveIEEE(fig, figDir, 'allmethods_metric_bar');
end

function plotClassBPSD(evalDir, figDir)
controllers = ["Passive","PID","SPDF","PPO","SI-RPPO"];
labels = ["Passive","PID","SPDF","PPO","SI-RPPO"];
fig = newFig('singleTall');
ax = axes(fig);
hold(ax, 'on');
for c = controllers
    T = readController(evalDir, c, 'class_b');
    if isempty(T), continue; end
    t = T.time;
    fs = 1 / median(diff(t));
    y = T.ddzb - mean(T.ddzb, 'omitnan');
    [pxx, f] = localPsd(y, fs);
    st = ctrlStyle(c);
    plot(ax, f, 10 * log10(pxx + eps), 'Color', st.Color, ...
        'LineStyle', st.LineStyle, 'LineWidth', st.LineWidth);
end
xlim(ax, [0, 25]);
xlabel(ax, 'Frequency (Hz)');
ylabel(ax, 'PSD (dB/Hz)');
legend(ax, labels, 'Location', 'northeast');
styleAxis(ax);
saveIEEE(fig, figDir, 'allmethods_psd');
end

function plotClassBAcceleration(evalDir, figDir)
controllers = ["Passive","PID","SPDF","PPO","SI-RPPO"];
labels = ["Passive","PID","SPDF","PPO","SI-RPPO"];
fig = newFig('singleTall');
[ax1, ax2] = stackedAxes(fig);
hold(ax1, 'on'); hold(ax2, 'on');
for c = controllers
    T = readController(evalDir, c, 'class_b');
    if isempty(T), continue; end
    idx = T.time <= min(8, max(T.time));
    st = ctrlStyle(c);
    plot(ax1, T.time(idx), T.ddzb(idx), 'Color', st.Color, ...
        'LineStyle', st.LineStyle, 'LineWidth', st.LineWidth);
    plot(ax2, T.time(idx), T.ddtheta(idx), 'Color', st.Color, ...
        'LineStyle', st.LineStyle, 'LineWidth', st.LineWidth);
end
ylabel(ax1, '$\ddot z_b$ (m/s$^2$)', 'Interpreter', 'latex');
ylabel(ax2, '$\ddot\theta$ (rad/s$^2$)', 'Interpreter', 'latex');
xlabel(ax2, 'Time (s)');
legend(ax1, labels, 'Location', 'northwest', 'NumColumns', 3);
styleAxis(ax1); styleAxis(ax2);
saveIEEE(fig, figDir, 'allmethods_ts_accel');
end

function plotClassBMotion(evalDir, figDir)
controllers = ["Passive","PID","SPDF","PPO","SI-RPPO"];
labels = ["Passive","PID","SPDF","PPO","SI-RPPO"];
fig = newFig('singleTall');
[ax1, ax2] = stackedAxes(fig);
hold(ax1, 'on'); hold(ax2, 'on');
for c = controllers
    T = readController(evalDir, c, 'class_b');
    if isempty(T), continue; end
    idx = T.time <= min(8, max(T.time));
    st = ctrlStyle(c);
    plot(ax1, T.time(idx), 1000 * T.zb(idx), 'Color', st.Color, ...
        'LineStyle', st.LineStyle, 'LineWidth', st.LineWidth);
    plot(ax2, T.time(idx), 1000 * T.theta(idx), 'Color', st.Color, ...
        'LineStyle', st.LineStyle, 'LineWidth', st.LineWidth);
end
ylabel(ax1, '$z_b$ (mm)', 'Interpreter', 'latex');
ylabel(ax2, '$\theta$ (mrad)', 'Interpreter', 'latex');
xlabel(ax2, 'Time (s)');
legend(ax1, labels, 'Location', 'northwest', 'NumColumns', 3);
styleAxis(ax1); styleAxis(ax2);
saveIEEE(fig, figDir, 'allmethods_ts_motion');
end

function plotComfortEnergy(metrics, figDir)
controllers = ["Passive","PID","SPDF","PPO","SI-RPPO"];
labels = ["被动","PID","SPDF","PPO","SI-RPPO"];
fig = newFig('singleTall');
ax = axes(fig);
hold(ax, 'on');
xVals = nan(numel(controllers), 1);
yVals = nan(numel(controllers), 1);
for i = 1:numel(controllers)
    c = controllers(i);
    rows = normalizeController(metrics.Controller) == c;
    if ~any(rows), continue; end
    x = mean(metrics.ControlAbsEnergy_J(rows), 'omitnan');
    y = mean(metrics.BodyAccRMS_mps2(rows), 'omitnan');
    xVals(i) = x;
    yVals(i) = y;
    st = ctrlStyle(c);
    scatter(ax, x, y, 40, 'MarkerFaceColor', st.Color, ...
        'MarkerEdgeColor', 'k', 'LineWidth', 0.35, ...
        'Marker', st.Marker);
    dx = 55;
    dy = 0.006;
    if c == "Passive"
        dx = 75;
        dy = -0.006;
    elseif c == "SPDF"
        dx = -410;
        dy = 0.010;
    elseif c == "PPO"
        dx = 75;
        dy = -0.010;
    end
    text(ax, x + dx, y + dy, char(labels(i)), ...
        'FontName', 'SimSun', 'FontSize', 7, ...
        'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
end
xlabel(ax, '控制能耗 (J)', 'FontName', 'SimSun');
ylabel(ax, '车身加速度 RMS (m/s^2)', 'FontName', 'SimSun');
xMax = max(xVals, [], 'omitnan');
xlim(ax, [0, xMax * 1.06]);
yMin = min(yVals, [], 'omitnan');
yMax = max(yVals, [], 'omitnan');
yPad = max(0.015, 0.10 * (yMax - yMin));
ylim(ax, [max(0, yMin - yPad), yMax + yPad]);
styleAxis(ax);
set(ax.XLabel, 'FontName', 'SimSun');
set(ax.YLabel, 'FontName', 'SimSun');
saveIEEE(fig, figDir, 'comfort_energy_tradeoff');
end

function plotGateBehavior(metrics, resultDir, evalDir, figDir)
metricsPath = fullfile(evalDir, 'metrics.csv');
M = readtable(metricsPath, 'TextType', 'string', 'VariableNamingRule', 'preserve');
rows = normalizeController(M.Controller) == "SI-RPPO";
scenarioLabels = M.ScenarioLabel(rows);
gate = M.PolicyImprovementGate(rows);
actionDelta = M.PolicyProjectionDeltaRMS_N(rows);

fig = newFig('singleTall');
[ax1, ax2] = stackedAxes(fig);
bar(ax1, gate, 'FaceColor', ctrlStyle("SI-RPPO").Color, ...
    'EdgeColor', 'k', 'LineWidth', 0.35);
bar(ax2, actionDelta, 'FaceColor', [0.35 0.35 0.35], ...
    'EdgeColor', 'k', 'LineWidth', 0.35);
set(ax1, 'XTick', 1:numel(scenarioLabels), 'XTickLabel', scenarioLabels);
set(ax2, 'XTick', 1:numel(scenarioLabels), 'XTickLabel', scenarioLabels);
set(ax1, 'XTickLabel', []);
xtickangle(ax2, 25);
ylabel(ax1, 'Gate');
ylabel(ax2, 'Projection RMS (N)');
xlabel(ax2, 'Scenario');
ylim(ax1, [0, 1.05]);
styleAxis(ax1); styleAxis(ax2);
saveIEEE(fig, figDir, 'sirppo_gate_behavior');
end

function plotTrainingCurves(resultDir, evalDir, figDir)
siPre = readJsonObject(fullfile(resultDir, '20260621_231449_ppo', 'imitation_pretrained_eval.json'));
metrics = readtable(fullfile(evalDir, 'metrics.csv'), ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
controllers = normalizeController(metrics.Controller);
siUnified = mean(metrics.EpisodeReturn(controllers == "SI-RPPO"), 'omitnan');
ppoUnified = mean(metrics.EpisodeReturn(controllers == "PPO"), 'omitnan');
returns = [fieldOr(siPre, 'mean_return', nan), siUnified, ppoUnified];
labels = categorical(["SI-RPPO BC","SI-RPPO unified","PPO unified"]);
labels = reordercats(labels, ["SI-RPPO BC","SI-RPPO unified","PPO unified"]);

fig = newFig('double');
ax = axes(fig, 'Position', [0.10 0.24 0.86 0.68]);
b = bar(ax, labels, returns, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 0.35);
b.CData = [ctrlStyle("SI-RPPO").Color; ctrlStyle("SI-RPPO").Color; ctrlStyle("PPO").Color];
ylabel(ax, 'Unified evaluation return');
xlabel(ax, 'Stage');
text(ax, 1:numel(returns), returns, compose('%.0f', returns), ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', 'FontSize', 6);
styleAxis(ax);
saveIEEE(fig, figDir, 'sirppo_training_curves');
end

function plotConvergence(resultDir, evalDir, figDir)
siPre = readJsonObject(fullfile(resultDir, '20260621_231449_ppo', 'imitation_pretrained_eval.json'));
metrics = readtable(fullfile(evalDir, 'metrics.csv'), ...
    'TextType', 'string', 'VariableNamingRule', 'preserve');
controllers = normalizeController(metrics.Controller);
siUnified = mean(metrics.EpisodeReturn(controllers == "SI-RPPO"), 'omitnan');
ppoUnified = mean(metrics.EpisodeReturn(controllers == "PPO"), 'omitnan');

fig = newFig('singleTall');
ax = axes(fig);
hold(ax, 'on');
plot(ax, [1, 2], [fieldOr(siPre, 'mean_return', nan), siUnified], ...
    'Color', ctrlStyle("SI-RPPO").Color, 'LineStyle', '-', 'Marker', 'o', 'LineWidth', 0.65);
plot(ax, 2, ppoUnified, ...
    'Color', ctrlStyle("PPO").Color, 'LineStyle', 'none', 'Marker', 'v', 'LineWidth', 0.65);
set(ax, 'XTick', [1 2], 'XTickLabel', {'BC','Unified'});
xlabel(ax, 'Stage');
ylabel(ax, 'Unified evaluation return');
legend(ax, ["SI-RPPO","PPO"], 'Location', 'southeast');
styleAxis(ax);
saveIEEE(fig, figDir, 'convergence_comparison');
end

function plotBumpResponse(evalDir, figDir)
controllers = ["Passive","PID","SPDF","PPO","SI-RPPO"];
labels = ["Passive","PID","SPDF","PPO","SI-RPPO"];
fig = newFig('singleTall');
[ax1, ax2] = stackedAxes(fig);
hold(ax1, 'on'); hold(ax2, 'on');
for c = controllers
    T = readController(evalDir, c, 'bump');
    if isempty(T), continue; end
    st = ctrlStyle(c);
    plot(ax1, T.time, T.ddzb, 'Color', st.Color, ...
        'LineStyle', st.LineStyle, 'LineWidth', st.LineWidth);
    plot(ax2, T.time, 1000 * T.zb, 'Color', st.Color, ...
        'LineStyle', st.LineStyle, 'LineWidth', st.LineWidth);
end
ylabel(ax1, '$\ddot z_b$ (m/s$^2$)', 'Interpreter', 'latex');
ylabel(ax2, '$z_b$ (mm)', 'Interpreter', 'latex');
xlabel(ax2, 'Time (s)');
legend(ax1, labels, 'Location', 'northwest', 'NumColumns', 3);
styleAxis(ax1); styleAxis(ax2);
saveIEEE(fig, figDir, 'bump_response');
end

function T = readController(evalDir, controller, scenario)
name = char(controller);
if controller == "Passive"
    name = 'PASSIVE';
end
path = fullfile(evalDir, sprintf('%s_%s.csv', name, scenario));
if ~exist(path, 'file')
    if controller == "PPO"
        legacyPath = fullfile(evalDir, sprintf('%s_%s.csv', 'PPO_scratch', scenario));
        if exist(legacyPath, 'file')
            path = legacyPath;
        else
            T = table();
            return;
        end
    else
        T = table();
        return;
    end
end
T = readtable(path, 'VariableNamingRule', 'preserve');
end

function C = normalizeController(C)
C = string(C);
C(upper(C) == "PASSIVE") = "Passive";
C(C == "passive") = "Passive";
C(C == "SI_RPPO") = "SI-RPPO";
C(C == "PPO_scratch") = "PPO";
end

function st = ctrlStyle(name)
switch string(name)
    case "Passive"
        st.Color = [0.000 0.447 0.741]; st.LineStyle = '-';  st.Marker = 'o';
    case "PID"
        st.Color = [0.000 0.000 0.000]; st.LineStyle = '-.'; st.Marker = 's';
    case "SPDF"
        st.Color = [0.850 0.120 0.100]; st.LineStyle = '-.'; st.Marker = '^';
    case "PPO"
        st.Color = [0.420 0.420 0.420]; st.LineStyle = '--'; st.Marker = 'v';
    case "TD3"
        st.Color = [0.929 0.694 0.125]; st.LineStyle = ':';  st.Marker = '>';
    case "SAC"
        st.Color = [0.301 0.745 0.933]; st.LineStyle = ':';  st.Marker = '<';
    case "DDPG"
        st.Color = [0.635 0.078 0.184]; st.LineStyle = ':';  st.Marker = 'p';
    case "SI-RPPO"
        st.Color = [0.000 0.500 0.220]; st.LineStyle = '-';  st.Marker = 'h';
    otherwise
        st.Color = [0.2 0.2 0.2]; st.LineStyle = '-'; st.Marker = 'o';
end
st.LineWidth = 0.55;
end

function fig = newFig(kind)
fig = figure('Visible', 'off', 'Color', 'w');
switch kind
    case 'double'
        fig.Units = 'inches';
        fig.Position = [1, 1, 7.16, 3.0];
    otherwise
        fig.Units = 'inches';
        fig.Position = [1, 1, 3.5, 2.2];
end
end

function [ax1, ax2] = stackedAxes(fig)
ax1 = axes(fig, 'Position', [0.16 0.57 0.80 0.35]);
ax2 = axes(fig, 'Position', [0.16 0.16 0.80 0.35]);
end

function [ax1, ax2, ax3] = threeAxes(fig)
ax1 = axes(fig, 'Position', [0.08 0.70 0.88 0.24]);
ax2 = axes(fig, 'Position', [0.08 0.39 0.88 0.24]);
ax3 = axes(fig, 'Position', [0.08 0.10 0.88 0.22]);
end

function styleAxis(ax)
set(ax, 'FontName', 'Times New Roman', 'FontSize', 7, ...
    'LineWidth', 0.5, 'Box', 'on', 'TickDir', 'in', ...
    'XMinorTick', 'off', 'YMinorTick', 'off');
grid(ax, 'on');
ax.GridLineStyle = ':';
ax.GridAlpha = 0.25;
if ~isempty(ax.Legend)
    ax.Legend.FontSize = 6;
    ax.Legend.Box = 'off';
end
end

function saveIEEE(fig, figDir, stem)
set(fig, 'PaperUnits', 'inches');
set(fig, 'PaperPositionMode', 'auto');
figSize = fig.Position(3:4);
set(fig, 'PaperSize', figSize, 'PaperPosition', [0 0 figSize]);
epsPath = fullfile(figDir, [stem, '.eps']);
pdfPath = fullfile(figDir, [stem, '.pdf']);
print(fig, epsPath, '-depsc', '-painters', '-r600');
print(fig, pdfPath, '-dpdf', '-painters', '-r600');
close(fig);
end

function H = readJsonArray(path)
if ~exist(path, 'file')
    H = struct([]);
    return;
end
H = jsondecode(fileread(path));
if isempty(H)
    H = struct([]);
elseif isstruct(H) && numel(H) == 1 && isfield(H, 'data')
    H = H.data;
end
end

function S = readJsonObject(path)
if ~exist(path, 'file')
    S = struct();
    return;
end
S = jsondecode(fileread(path));
end

function value = fieldOr(S, fieldName, defaultValue)
value = defaultValue;
if isstruct(S) && isfield(S, fieldName)
    v = S.(fieldName);
    if isnumeric(v) && isscalar(v)
        value = v;
    end
end
end

function values = numericField(S, fieldName)
values = [];
if isempty(S) || ~isstruct(S)
    return;
end
values = nan(1, numel(S));
for i = 1:numel(S)
    if isfield(S(i), fieldName)
        v = S(i).(fieldName);
        if isnumeric(v) && isscalar(v)
            values(i) = v;
        end
    end
end
mask = ~isnan(values);
values = values(mask);
end

function [pxx, f] = localPsd(y, fs)
y = y(:);
if exist('pwelch', 'file') == 2
    nwin = min(512, max(64, 2 ^ floor(log2(numel(y) / 4))));
    [pxx, f] = pwelch(y, hann(nwin), floor(nwin / 2), nwin, fs);
else
    n = numel(y);
    Y = fft(y .* hann(n));
    p2 = abs(Y / n) .^ 2;
    pxx = p2(1:floor(n / 2) + 1);
    pxx(2:end-1) = 2 * pxx(2:end-1);
    f = fs * (0:floor(n / 2))' / n;
end
end
