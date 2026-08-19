function write_road_run_par(template_par, output_par, station, elevation, road_label)
%WRITE_ROAD_RUN_PAR Replace the first longitudinal elevation table in Run_all.par.
arguments
    template_par (1,:) char
    output_par (1,:) char
    station (:,1) double
    elevation (:,1) double
    road_label (1,:) char
end

assert(isfile(template_par), 'CarSim run parameter file not found: %s', template_par);
assert(numel(station) == numel(elevation) && numel(station) >= 2, ...
    'Road station and elevation arrays must have equal length and at least two rows.');
assert(all(isfinite(station)) && all(isfinite(elevation)), ...
    'Road station and elevation values must be finite.');
assert(all(diff(station) > 0), 'Road station must be strictly increasing.');

text = fileread(template_par);
[table_start, table_end] = regexp(text, ...
    '(?m)^ROAD_DZ_TABLE[^\r\n]*\r?\n[\s\S]*?^ENDTABLE[^\r\n]*', ...
    'start', 'end', 'once');
assert(~isempty(table_start), 'ROAD_DZ_TABLE was not found in %s', template_par);

if contains(text, sprintf('\r\n'))
    line_break = sprintf('\r\n');
else
    line_break = sprintf('\n');
end
rows = compose('%.6f, %.12g', station, elevation);
replacement = ['ROAD_DZ_TABLE LINEAR_FLAT', line_break, ...
    strjoin(cellstr(rows), line_break), line_break, 'ENDTABLE'];
text = [text(1:table_start-1), replacement, text(table_end+1:end)];
text = strrep(text, 'Class C #1', road_label);

output_dir = fileparts(output_par);
if ~isfolder(output_dir), mkdir(output_dir); end
fid = fopen(output_par, 'w');
assert(fid >= 0, 'Cannot write %s', output_par);
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
count = fwrite(fid, text, 'char');
assert(count == numel(text), 'Incomplete write to %s', output_par);
end
