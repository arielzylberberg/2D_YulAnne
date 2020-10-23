init_path;

%% Compute ITI
Fit.RewardRate.get_intertrial_interval;

%% Load ITI
L_data = load('../Data_2D/Fit.RewardRate/get_intertrial_interval');

%% Compute expected reward rate from 


%% Compute expected and actual reward rate
th_kind = 'min_sub';
switch th_kind
    case 'min_sub'
        file = '../Data_2D/Fit.RewardRate.main_reward_rate/pred_data_by_model_w_th_min_sub';
    case 'fit'
        file = '../Data_2D/Fit.RewardRate.main_reward_rate/pred_data_by_model';
end

L_data_pred = load(file);
datas = L_data_pred.data;
preds = L_data_pred.pred;
models = L_data_pred.models;
n_model1 = size(preds, 2);
n_boot = 100;

mean_iti = L_data.mean_iti;

%%
res = struct;
res.reward_rate = zeros(n_subj, n_model1, n_boot);
res.mean_RT = zeros(n_subj, n_model1, n_boot);
res.mean_accu = zeros(n_subj, n_model1, n_boot);
info = cell(n_subj, n_model1);

n_cond = sizes(preds{1,1}, [2, 3]);

for i_subj = 1:n_subj
    for i_model = 1:n_model1 % [1, 4, n_model1] % 
        data = datas{i_subj};
        n_tr_cond_data = sums(data, [1, 4, 5]);
        
        for i_boot = 1:n_boot
            pred = preds{i_subj, i_model};
            
            if isempty(pred)
                continue;
            end
                
            if i_boot > 1
                pred = Fit.D2.bootstrap(pred, data, i_boot);
                pred = bsxfun(@rdivide, pred, n_tr_cond_data);
            end
            
            [reward_rate1, mean_RT1, mean_accu1, info1] = ...
                Fit.RewardRate.get_reward_rate(data, pred, t, ...
                    mean_iti(i_subj));

            res.reward_rate(i_subj, i_model, i_boot) = reward_rate1;        
            res.mean_RT(i_subj, i_model, i_boot) = mean_RT1;
            res.mean_accu(i_subj, i_model, i_boot) = mean_accu1;
            info{i_subj, i_model} = info1;
        end
    end
end

%% Calculate CI and save
est = struct;
err = struct;
for f = {'reward_rate', 'mean_RT', 'mean_accu'}
    est.(f{1}) = mean(res.(f{1}), 3);
    err.(f{1}) = std(res.(f{1}), 0, 3);
end

file = ['../Data_2D/Fit.RewardRate.main_reward_rate/reward_rate_boot+th=' th_kind];
mkdir2(fileparts(file));
save(file, 'res', 'est', 'err', 'info', 'models', 'mdl_disp_names');
fprintf('Saved bootstrapped reward rate to %s\n', file);

%% Load reward rate
file = ['../Data_2D/Fit.RewardRate.main_reward_rate/reward_rate_boot+th=' th_kind];
load(file, 'res', 'est', 'err', 'info', 'models', 'mdl_disp_names');
subjs = Data.Consts.subjs_RT;
n_subj = numel(subjs);
n_model = numel(models);

%% Save csv
for f = fieldnames(est)'
    ds_abs = dataset;
    ds_abs.subj = subjs(:);

    ds_rel = dataset;
    ds_rel.subj = subjs(:);

    for i_model = 1:n_model
    	model = models{i_model};
        for i_subj = 1:n_subj
            ds_abs.(model){i_subj,1} = ...
                sprintf('%1.3f +- %1.3f', ...
                    est.(f{1})(i_subj, i_model), ...
                    err.(f{1})(i_subj, i_model));
                
            est_data = est.(f{1})(i_subj, end);
            ds_rel.(model){i_subj,1} = ...
                sprintf('%1.1f%% +- %1.1f%%', ...
                    est.(f{1})(i_subj, i_model) / est_data * 100, ...
                    err.(f{1})(i_subj, i_model) / est_data * 100);
        end
    end
    disp(ds_abs);
    disp(ds_rel);
    pth = '../Data_2D/Fit.RewardRate.main_reward_rate';
    file = fullfile(pth, sprintf('%s_abs+th=%s.csv', f{1}, th_kind));
    export(ds_abs, 'file', file, 'Delimiter', ',');
    fprintf('Saved to %s\n', file);

    file = fullfile(pth, sprintf('%s_rel+th=%s.csv', f{1}, th_kind));
    export(ds_rel, 'file', file, 'Delimiter', ',');
    fprintf('Saved to %s\n', file);
end

%%
fs = {
    'reward_rate', 'Reward Rate'
    'mean_RT', 'Mean RT'
    'mean_accu', 'Accuracy'
    }';
for f1 = fs
    [f, name] = deal(f1{:});
    fig_tag(f);
    clf;
    
    ax = gobjects(1, n_subj);
    for i_subj = 1:n_subj
        ax(i_subj) = subplot(1, n_subj, i_subj);

        est1 = est.(f)(i_subj,:);
        est_orig = est1(end);
        est_rel = est1(1:(end-1)) / est_orig * 100;
        
        err1 = err.(f)(i_subj,:);
        err_rel = err1(1:(end-1)) / est_orig * 100;
        
        n_model = numel(est_rel);
        barh(est_rel, 'w');
        set(gca, 'YDir', 'reverse');
        
        x_min = min(est_rel(:) - err_rel(:));
        x_max = max(est_rel(:) + err_rel(:));
        x_dif = x_max - x_min;        
        xlim([x_min - x_dif/20, x_max + x_dif/20]);
        
        ylim([0.5, n_model + 0.5]);
        bml.plot.beautify;
        h = crossLine('v', 100, {':', [0.7, 0.7, 0.7]});
        uistack(h, 'top');
        set(h, 'LineWidth', 2);
        
        lb1 = est_rel - err_rel;
        ub1 = est_rel + err_rel;
        y = 1:n_model;
        hold on;
        plot([lb1; ub1], [y; y], 'k-', 'LineWidth', 2);
        hold off;
        
        if i_subj == 1
            set(gca, 'YTickLabel', mdl_disp_names(1:(end-1),2));
        else
            set(gca, 'YTickLabel', '');
        end
        if i_subj == 2
            title(sprintf(['Subject\n\n', subjs{i_subj}]));
            xlabel(sprintf('Relative %s (%%)', name));
        else
            title(subjs{i_subj});
        end
    end
    %%
    bml.plot.position_subplots(ax, ...
        'margin_left', 0.25, ...
        'margin_right', 0.01, ...
        'margin_top', 0.2, ...
        'margin_bottom', 0.16);
    file = fullfile('../Data_2D/Fit.RewardRate.main_reward_rate', [f '+th=' th_kind]);
    savefigs(file, 'size', [400, 200]);
end