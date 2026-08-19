function [station, elevation] = extract_carsim_road_table(par_file)
%EXTRACT_CARSIM_ROAD_TABLE Read the first ROAD_DZ_TABLE from a CarSim PAR file.
arguments
    par_file (1,:) char
end
assert(isfile(par_file), 'CarSim run parameter file not found: %s', par_file);
lines = readlines(par_file);
start_idx = find(startsWith(strtrim(lines), "ROAD_DZ_TABLE "), 1, 'first');
assert(~isempty(start_idx), 'ROAD_DZ_TABLE was not found in %s', par_file);
end_rel = find(strtrim(lines(start_idx+1:end)) == "ENDTABLE", 1, 'first');
assert(~isempty(end_rel), 'ROAD_DZ_TABLE in %s is not terminated.', par_file);
rows = lines(start_idx+1:start_idx+end_rel-1);
station = zeros(numel(rows),1); elevation = zeros(numel(rows),1);
count = 0;
for i = 1:numel(rows)
    token = regexp(char(rows(i)), ...
        '^\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)', 'tokens', 'once');
    if isempty(token), continue; end
    count = count + 1;
    station(count) = str2double(token{1});
    elevation(count) = str2double(token{2});
end
station = station(1:count); elevation = elevation(1:count);
assert(count >= 2 && all(diff(station) > 0), ...
    'Extracted CarSim road table is empty or not strictly increasing.');
end
