function get_intertrial_interval
%%
init_path;

subjs = Data.Consts.subjs_RT;
n_subj = numel(subjs);
parad = 'RT';
Ls = cell(n_subj, 1);

%% Load data
for i_subj = 1:n_subj
    subj = subjs{i_subj};
    file = sprintf('%s_%s', parad, subj);
    Ls{i_subj} = load(fullfile('../Data_2D/sTr', file));
end

%% Compute mean ITI
tr_st_sec = cell(n_subj, 1);
iti = cell(n_subj, 1);
rt = cell(n_subj, 1);
cond = cell(n_subj, 1);
mean_iti = zeros(n_subj, 1);
end_of_run = cell(n_subj, 1);
accu_all = cell(n_subj, 1);
accu = cell(n_subj, 1);

for i_subj = 1:n_subj
    dat = Ls{i_subj}.dat;
    
    tr_st_sec1 = dat.timestamp * 24 * 3600;
    iti1 = [diff(tr_st_sec1); 0] - dat.RT;
    
    end_of_run1 = [diff(dat.i_all_Run) > 0; true];
    iti1(end_of_run1) = 0;
    
    tr_st_sec{i_subj} = tr_st_sec1;
    iti{i_subj} = iti1;
    rt{i_subj} = dat.RT;
    cond{i_subj} = [dat.condM, dat.condC];
    end_of_run{i_subj} = end_of_run1;
    accu_all{i_subj} = [dat.subjM == dat.corrM, dat.subjC == dat.corrC];
    accu{i_subj} = all(accu_all{i_subj}, 2);
    
    mean_iti(i_subj) = nanmean(iti1);
end

%% Save interim data
clear Ls dat

file = '../Data_2D/Fit.RewardRate/get_intertrial_interval';
mkdir2(fileparts(file));
save(file);
fprintf('Saved to %s\n', file);