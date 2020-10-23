init_path;

%%
dat = dataset;
parads = {'RT', 'sh'};
for parad = parads
    subjs = Data.Consts.(['subjs_' parad{1}]);
    n_subj = numel(subjs);
    for i_subj = 1:n_subj
        file = fullfile('../../Data_2D/sTr', ...
            sprintf('%s_%s.mat', parad{1}, subjs{i_subj}));
        L = load(file);
        n_tr = size(dat, 1);
        n_new = size(L.dat, 1);
        fprintf('Loaded %d trials from %s\n', n_new, file);

        ix_new = n_tr + (1:n_new);
        L.dat.i_subj = zeros(n_new, 1) + i_subj;
        L.dat.parad_short = repmat(parad(1), [n_new, 1]);
        L.dat.parad = L.dat.parad_short;
        L.dat.i_all_sTr = (1:n_new)';

        dat = bml.ds.ds_set(dat, ix_new, L.dat);
    end
end

%%
file_combined = '../../Data_2D/sTr/combined_2D_RT_sh.mat';
save(file_combined, 'dat');
fprintf('Saved to %s\n', file_combined);
