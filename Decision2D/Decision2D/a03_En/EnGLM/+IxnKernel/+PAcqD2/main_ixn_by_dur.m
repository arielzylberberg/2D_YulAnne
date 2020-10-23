clear;
init_path;

%% == Load 2D data
subjs = Data.Consts.subjs_RT(1:3); % (1:3); % (4); % 
n_subj = numel(subjs);
parad = 'RT'; % 'sh'; % 

dats0 = cell(n_subj, 1);
for i_subj = 1:n_subj
    subj = subjs{i_subj};
    file_dat = fullfile('../Data_2D/sTr', sprintf('%s_%s.mat', ...
        parad, subj));
    L0 = load(file_dat);
    dats0{i_subj} = L0.dat;
    fprintf('Loaded %s\n', file_dat);
end

%% == Filter data
dif_irr_easy = [-1, 0];
dif_irr_hard = 1:2; % :2;

% dif_rel_incls{dim}{subj}
dif_rel_incls = {{1:2, 1:2, 1}, {1:2, 1:2, 1:2}};

%%
for i_subj = 1:n_subj
    for dim = 1:n_dim
    end
end