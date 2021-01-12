% Count the number of days subjects actually performed the task.

for parad = {'RT', 'sh'}
    for subj = {'S1', 'S2', 'S3'}
        file = sprintf('../../Data_2D/sTr/%s_%s.mat', parad{1}, subj{1});
        L = load(file);
        fprintf('Loaded %s\n', file);
        dates = arrayfun(@(v) datestr(v, 'yyyymmdd'), L.dat.timestamp, ...
            'UniformOutput', false);
        fprintf('%s, %s: %d days\n', ...
            parad{1}, subj{1}, numel(unique(dates)));
    end
end