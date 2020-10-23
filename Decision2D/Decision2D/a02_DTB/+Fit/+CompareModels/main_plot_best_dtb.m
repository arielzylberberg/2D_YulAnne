init_path;

%% Load/plot best
L = load('/Users/yulkang/Dropbox/CodeNData_2D/ExtRepos/ShadlenLab/Decision2D/Data_2D/Fit.main_compare_dtb_all/main_compare_dtb_all.mat', ...
    'ds_best');
ds_best = L.ds_best;
files = ds_best.file;

subjs = Data.Consts.subjs_RT(1:3);
n_subj = numel(subjs);
Ls = cell(n_subj, 1);
Ws = cell(n_subj, 1);

for i_subj = 1:n_subj
    subj = subjs{i_subj};
    L1 = load(files{i_subj}, 'Fl');
    L1.Fl.res2W;
    Ls{i_subj} = L1;
    Ws{i_subj} = L1.Fl.W;
end

%% Average across subjects
datas = cell(1, n_subj);
preds = cell(1, n_subj);

% n_dim = Data.Consts.n_dim;
% preds_oversampled = cell(n_dim, n_subj);
% oversample_factor = 10;

for i_subj = 1:n_subj
    if isempty(Ws{i_subj})
        continue;
    end
    W = Ws{i_subj};
    W.force_repeated_pred = true;
    
    datas{i_subj} = W.Data.RT_data_pdf;
%     datas{i_subj} = datas{i_subj} ./ sum(datas{i_subj}(:));

    preds{i_subj} = W.Data.RT_pred_pdf;
%     preds{i_subj} = preds{i_subj} ./ sum(preds{i_subj}(:));

%     for dim = 1 % :n_dim
%         oversample_factors = ones(1, n_dim);
%         oversample_factors(dim) = oversample_factor;
%         W.Data.set_conds_oversample_factor(oversample_factors);
%         W.pred;
%         preds_oversampled{dim, i_subj} = W.Data.RT_pred_pdf;        
%     end

%     oversample_factors = ones(1, n_dim);
%     W.Data.set_conds_oversample_factor(oversample_factors);
%     W.pred;
end

%%
data = sum(cat(6, datas{:}), 6);
pred = mean(cat(6, preds{:}), 6);

% W1 = W.deep_copy;
% W1.Data.subj = 'S0';
% W1.Data.RT_data_pdf_ = data;
% W1.Data.RT_pred_pdf_ = pred;
% W1.Data.loaded = true;

%%
C = varargin2C({
    'dimOnX', 1
    });

Pl = DtbPlot.PlotRt2D;
Pl.plot_p(pred, 'src', 'pred', C{:});
hold on;

Pl = DtbPlot.PlotRt2D;
Pl.plot_p(data, 'src', 'data', C{:});
hold off;

%%
yfun = 'mean';
for dim_on_x = 1:2
    W = Fit.D2.Bounded.Main;
    C = varargin2C({
        'dim_on_x', dim_on_x
        'group_y', {4:5, 1:3}
        'yfun', yfun
        'p_data', data
        'p_pred', pred
        'to_use_to_excl', false % Set false to use all data
        });
    
    clf;
    [h_pred, h_data] = W.plot_rt_vs_rt(C{:});
    
    switch dim_on_x
        case 1
            ticks = 0:0.5:5;
        case 2
            ticks = 0:0.2:5;
    end
    set(gca, 'XTick', ticks, 'YTick', ticks);
    set(h_pred{1}, 'LineWidth', 1);
    set(h_data{1}, 'LineWidth', 0.5, 'MarkerSize', 5);
    
%     W.plot_rt_vs_rt_unit(pred, 'style', 'pred', C{:});
%     hold on;
% 
%     [h, hxe, hye, y, e] = W.plot_rt_vs_rt_unit(data, 'style', 'data', C{:});
%     hold off;
% 
%     bml.plot.beautify;
    
    pth = '../Data_2D/Fit.CompareModels.main_plot_best_dtb';
    S_file = varargin2S({
        'sbj', subjs
        'plt', 'rt_vs_rt'
        'dmr', dim_on_x
        'yfun', yfun
        });
    S2s = bml.str.Serializer;
    nam = S2s.convert(S_file);
    file = fullfile(pth, nam);
    
    ax = gca;
    bml.plot.position_subplots(ax, ...
        'margin_top', 0.01, 'margin_right', 0.01, ...
        'margin_left', 0.4, 'margin_bottom', 0.4);
    savefigs(file, 'size', [100, 100]);
end


%% Best model
% pth = 'Fit.CompareModels.main_plot_best_dtb';
% for i_subj = 1:n_subj
%     W = Ws{i_subj};
%     W.plot_and_save_all('subdir', pth, ...
%         'remove_fields', setdiff(fieldnames(W.S_file), ...
%             {'sbj', 'prd', 'tsk', 'mdl'}));
% %     W.save_mat;
% end
% 
% %% Combine plots across subjects & models
% L = load('/Users/yulkang/Dropbox/CodeNData_2D/ExtRepos/ShadlenLab/Decision2D/Data_2D/Fit.main_compare_dtb_all/main_compare_dtb_all.mat');
% files = dataset2cell(L.ds_file(:,2:end));
% models = files(1,:);
% files = files(2:end,:);
% n_model = numel(models);
% 
% for i_subj = 1:n_subj
%     for i_model = 1:n_model
%         L1 = load(files{i_subj, i_model});
%         
%         L1.Fl.res2W;
%         W = L1.Fl.W;
%         W.plot_and_save_all('subdir', pth, ...
%             'add_fields', {'mdl', models{i_model}}, ...
%             'remove_fields', setdiff(fieldnames(W.S_file), ...
%                 {'sbj', 'prd', 'tsk'}));
%     end
% end