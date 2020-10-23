init_path;

%% Load models
L_comp = load('../Data/Fit.main_compare_dtb_all/main_compare_dtb_all.mat');

ds_file = L_comp.ds_file;
n_model = size(L_comp.mdl_disp_names, 1);
n_model1 = n_model + 1;
mdl_disp_names = [
    L_comp.mdl_disp_names
    {'data', 'Data'}];
models = mdl_disp_names(:,1);
subjs = L_comp.subjs;
n_subj = numel(subjs);

data = cell(n_subj, 1);
pred = cell(n_subj, n_model1);
cond_ch_incl_train = cell(n_subj, 1);
cond_ch_incl_valid = cell(n_subj, 1);
S0_file = cell(n_subj, n_model1);
S_file = cell(n_subj, n_model1);

%%
Ls = cell(n_subj, n_model);
for i_subj = 1:n_subj
    for i_model = 1:n_model1
        if i_model == n_model1 % use data
            L = Ls{i_subj, 1};
            W = L.Fl.W;
            t = W.t(:);
            pred1 = W.Data.RT_data_pdf;
            pred1 = bsxfun(@rdivide, pred1, sums(pred1, [1, 4, 5]));
        else
            model = models{i_model};
            file1 = ds_file.(model){i_subj};
            fprintf('Loading S%d/%d, model %d/%d: %s\n', ...
                i_subj, n_subj, i_model, n_model, file1);
            L1 = load(file1);
            L1.Fl.res2W;
            Ls{i_subj, i_model} = L1;

            L = Ls{i_subj, i_model};
            W = L.Fl.W;
            t = W.t(:);
            pred1 = W.Data.RT_pred_pdf;
        end
        data1 = W.Data.RT_data_pdf;        
        data{i_subj} = data1;
        pred{i_subj, i_model} = pred1;
        cond_ch_incl_train{i_subj} = W.get_cond_ch_to_include_train;
        cond_ch_incl_valid{i_subj} = W.get_cond_ch_to_include_train;
        S0_file{i_subj, i_model} = W.S0_file;
        S_file{i_subj, i_model} = W.S_file;
    end
end

%%
file = '../Data/Fit.RewardRate.main_reward_rate/pred_data_by_model';
mkdir2(fileparts(file));
save(file, 'models', 'mdl_disp_names', ...
    'ds_file', 'data', 'pred', 'subjs', ...
    'cond_ch_incl_train', 'cond_ch_incl_valid', ...
    'S0_file', 'S_file');
fprintf('Saved pred and data to %s\n', file);
