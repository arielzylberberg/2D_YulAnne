clear;
init_path;

%% Load/plot data, Ser, and Par
L = load('../../Data_2D/Fit.main_compare_dtb_all/main_compare_dtb_all.mat');

%%
models = {'mdl_Ser', 'mdl_Par'};
n_model = numel(models);

ds_file = L.ds_file(:, models);

subjs = Data.Consts.subjs_RT(1:3);
n_subj = numel(subjs);
Ls = cell(n_subj, n_model);
Ws = cell(n_subj, n_model);

for i_subj = 1:n_subj
    for i_model = 1:n_model
        model1 = models{i_model};
        file1 = ds_file.(model1){i_subj};
        
        fprintf('Loading %s\n', file1);
        file1 = strrep(file1, '../Data/', '../../Data_2D/');
        L1 = load(file1, 'Fl');
        L1.Fl.res2W;
        Ls{i_subj, i_model} = L1;
        Ws{i_subj, i_model} = L1.Fl.W;
    end
end
    
%% Average across subjects
datas = cell(n_subj);
preds = cell(n_subj, n_model);

% n_dim = Data.Consts.n_dim;
% preds_oversampled = cell(n_dim, n_subj);
% oversample_factor = 10;

for i_subj = 1:n_subj
    if isempty(Ws{i_subj})
        continue;
    end
    W = Ws{i_subj,1};
    W.force_repeated_pred = true;
    
    datas{i_subj} = W.Data.RT_data_pdf;
%     datas{i_subj} = datas{i_subj} ./ sum(datas{i_subj}(:));

    for i_model = 1:n_model
        W = Ws{i_subj, i_model};
        preds{i_subj, i_model} = W.Data.RT_pred_pdf;
%     preds{i_subj} = preds{i_subj} ./ sum(preds{i_subj}(:));
    end

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
pred = cat(6, preds{:});
pred = permute(reshape(pred, [size(data), [n_subj, n_model]]), ...
    [1, 2, 3, 4, 5, 7, 6]);
pred = mean(pred, 7);

%% Ch & RT plot
f_colormap = @(n_series) {
    flipud(bml.plot.colormaps.winter2(n_series))
    flipud(bml.plot.colormaps.cool2(n_series))
    };
n_data = 3;
colormaps = f_colormap(n_data);
model_to_plot = 1; % serial
n_dim = 2;
xlabels = {'Relative motion strength', 'Relative color strength'};

n_plot = 2;
clf;
ax = subplotRCs(n_plot, n_dim);

for dim_on_x = 1:n_dim
    C = varargin2C({
        'dimOnX', dim_on_x
        });
    dim_series = n_dim + 1 - dim_on_x;

    % Choice
    ax1 = ax(1, dim_on_x);
    axes(ax1);
    cla;
    Pl = DtbPlot.PlotCh2D;

    pred1 = pred(:,:,:,:,:,model_to_plot);
%     pred1 = Pl.group_p(pred(:,:,:,:,:,model_to_plot), dim_series, ...
%         'group', [1, 2, 2, 3, 3, 3, 2, 2, 1]);
    
    h_pred1 = Pl.plot_p(pred1, 'src', 'pred', ...
        'groupAxis', {[], [1, 1, 2, 2, 3]}, ...
        C{:});
%     h_pred = Pl.plot_p(pred1, 'src', 'pred', C{:});
    hold on;
    set(h_pred1, 'LineStyle', '-', 'LineWidth', 1);

    pred1 = pred(:,:,:,:,:,2);
%     pred1 = Pl.group_p(pred(:,:,:,:,:,model_to_plot), dim_series, ...
%         'group', [1, 2, 2, 3, 3, 3, 2, 2, 1]);
    
    h_pred2 = Pl.plot_p(pred1, 'src', 'pred', ...
        'groupAxis', {[], [1, 1, 2, 2, 3]}, ...
        C{:});
%     h_pred = Pl.plot_p(pred1, 'src', 'pred', C{:});
    hold on;
    set(h_pred2, 'LineStyle', '--', 'LineWidth', 1);

    Pl = DtbPlot.PlotCh2D;
%     Pl.groupAxis{dim_series} = [1, 1, 2, 2, 3];
%     data1 = Pl.group_p(data, dim_series, ...
%         'group', [1, 2, 2, 3, 3, 3, 2, 2, 1]);
%     h_data = Pl.plot_p(data1, 'src', 'data', C{:});
    h_data = Pl.plot_p(data, 'src', 'data', ...
        'groupAxis', {[], [1, 1, 2, 2, 3]}, ...
        C{:});
    hold off;
    
    for jj = 1:n_data % length(h_pred)
        set(h_pred1(jj), 'Color', colormaps{dim_series}(jj,:));
        set(h_pred2(jj), 'Color', colormaps{dim_series}(jj,:));
        set(h_data(jj), 'MarkerFaceColor', colormaps{dim_series}(jj,:), ...
            'MarkerSize', 5);
    end
    
    xlim([-1.1, 1.1]);
    axle1 = get(gca);
    xlabel('');
    set(ax1, 'XTickLabel', {'', '', '', '', ''});
%     set(ax1, 'YTickLabel', {'0', '', '', '', '1'});
    if dim_on_x == 2
        set(ax1, 'YTickLabel', '');
    end
    
    % RT
    ax1 = ax(2, dim_on_x);
    axes(ax1);
    cla;
    pred1 = pred(:,:,:,:,:,model_to_plot);
    Pl = DtbPlot.PlotRt2D;    
    h_pred1 = Pl.plot_p(pred1, 'src', 'pred', ...
        'groupAxis', {[], [1, 1, 2, 2, 3]}, C{:});
    set(h_pred1, 'LineStyle', '-', 'LineWidth', 1);
    hold on;

    pred1 = pred(:,:,:,:,:,2);
    Pl = DtbPlot.PlotRt2D;    
    h_pred2 = Pl.plot_p(pred1, 'src', 'pred', ...
        'groupAxis', {[], [1, 1, 2, 2, 3]}, C{:});
    set(h_pred2, 'LineStyle', '--', 'LineWidth', 1);
    hold on;

    Pl = DtbPlot.PlotRt2D;
    h_data = Pl.plot_p(data, 'src', 'data', ...
        'groupAxis', {[], [1, 1, 2, 2, 3]}, C{:});
    hold off;
    
    for jj = 1:n_data % size(h_pred, 1)
        set(h_pred1(jj,:), 'Color', colormaps{dim_series}(jj,:));
        set(h_pred2(jj,:), 'Color', colormaps{dim_series}(jj,:));
        set(h_data(jj,:), 'MarkerFaceColor', colormaps{dim_series}(jj,:), ...
            'MarkerSize', 5);
    end
    
    xlim([-1.1, 1.1]);
    xlabel(xlabels{dim_on_x});
    set(ax1, 'XTickLabel', {'-1', '', '0', '', '1'});
    if dim_on_x == 2
        set(ax1, 'YTickLabel', '');
        ylabel('');
    end    
end

for i_plot = 1:n_plot
    sameAxes(ax(i_plot,:), [], [], 'y');
end
bml.plot.position_subplots(ax, ...
    'margin_left', 0.15, ...
    'margin_right', 0.05, ...
    'margin_bottom', 0.17, ...
    'margin_top', 0.05, ...
    'btw_row', 0.05, ...
    'btw_col', 0.1);
% drawnow;
% 
% %%
% for i_plot = 1:numel(ax)
%     ax1 = ax(i_plot);
%     axis1 = get(ax1, 'XAxis');
%     axis1.Axle.VertexData(1,:) = [-1, 1];
% end
% drawnow;

pth = '../Data_2D/Fit.CompareModels.main_plot_pooled_ser_vs_par';
S_file = varargin2S({
    'sbj', subjs
    'plt', 'ch_rt'
    });
S2s = bml.str.Serializer;
nam = S2s.convert(S_file);
file = fullfile(pth, nam);
savefigs(file, 'size', [280, 200]);

%% RT-RT plot
yfun = 'mean';
line_styles = {'-', '--'};
n_data = 5;
colormaps = f_colormap(n_data);
% colormaps = {'winter', 'cool'};
% colormap1 = cool(5);
% colormaps = {cool(5), flipud(winter(5))};
% colormap1 = flipud(winter(5));
% colormaps = {colormap1(:,[2,3,1]), colormap1};

for dim_on_x = 1:2
    W = Fit.D2.Bounded.Main;
    clf;
    
    for i_model = 1:n_model
        pred1 = pred(:,:,:,:,:,i_model);
    
        C = varargin2C({
            'dim_on_x', dim_on_x
            'group_y', {4:5, 1:3}
            'yfun', yfun
            'p_data', data
            'p_pred', pred1
            'to_use_to_excl', false % Set false to use all data
            });

        [h_pred, h_data] = W.plot_rt_vs_rt(C{:});

        set(h_pred{1}, 'LineStyle', line_styles{i_model});
        set(h_pred{1}, 'LineWidth', 1);
        set(h_data{1}, 'LineWidth', 0.5, 'MarkerSize', 5);
        
        n_data = numel(h_data{1});
        colors = colormaps{dim_on_x};
%         colors = flipud(feval(colormaps{dim_on_x}, n_data)) ;
        for i_data = 1:n_data
            set(h_data{1}(i_data), 'MarkerFaceColor', colors(i_data,:));
        end
        hold on;

    %     W.plot_rt_vs_rt_unit(pred, 'style', 'pred', C{:});
    %     hold on;
    % 
    %     [h, hxe, hye, y, e] = W.plot_rt_vs_rt_unit(data, 'style', 'data', C{:});
    %     hold off;
    % 
    %     bml.plot.beautify;
    end
    hold off;
    switch dim_on_x
        case 1
            ticks = 0:0.5:5;
        case 2
            ticks = 0:0.2:5;
    end
    set(gca, 'XTick', ticks, 'YTick', ticks);

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
        'margin_top', 0.02, 'margin_right', 0.02, ...
        'margin_left', 0.4, 'margin_bottom', 0.4);
    savefigs(file, 'size', [100, 100]);
end

vars = bml.file.vars_wo_figs;
pth = '../Data_2D/Fit.CompareModels.main_plot_pooled_ser_vs_par';
file = fullfile(pth, 'ws_pooled_RT');
save(file, vars.name);
fprintf('Saved workspace to ws_pooled_RT.mat\n');

%%

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