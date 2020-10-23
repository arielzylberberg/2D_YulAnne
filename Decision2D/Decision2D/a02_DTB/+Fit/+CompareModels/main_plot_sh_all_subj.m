clear;
init_path;

%% short paradigm
subjs = {'S1', 'S2', 'S3'};
S0_file = varargin2S({
    'sbj', subjs
    'dfr', 1:4 % 1:4 | []
    });

% subjs = {'FR'};
% S0_file = varargin2S({
%     'sbj', subjs
%     'dfr', 1:6
%     });

if isequal(subjs, {'FR'})
    assert(isequal(S0_file.dfr, 1:6));
    files = {
        '../Data_2D/Fit.D2.RT.Td2Tnd.Main/sbj=FR+prd=RT+tsk=A+dtk=2+dmr=1+trm=1+eor=0+msf=0+ef=1+ec=-1+lf=0+eb=10+td=Par+smr=NaN+um=5+us=5.mat'
        '../Data_2D/Fit.D2.RT.Td2Tnd.Main/sbj=FR+prd=RT+tsk=A+dtk=2+dmr=1+trm=1+eor=0+msf=0+ef=1+ec=-1+lf=0+eb=10+td=Ser+smr=NaN+um=5+us=5.mat'
        };
else % humans
    if isequal(S0_file.dfr, 1:4)
        files = {
            '../Data_2D/Fit.D2.RT.Td2Tnd.Main/sbj=S1+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=0+msf=0+ef=1+ec=-1+lf=0+eb=10+td=Par+smr=NaN+um=2+us=2.mat'
            '../Data_2D/Fit.D2.RT.Td2Tnd.Main/sbj=S1+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=0+msf=0+ef=1+ec=-1+lf=0+eb=10+td=Ser+smr=NaN+um=2+us=2.mat'
            '../Data_2D/Fit.D2.RT.Td2Tnd.Main/sbj=S2+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=0+msf=0+ef=1+ec=-1+lf=0+eb=10+td=Par+smr=NaN+um=2+us=2.mat'
            '../Data_2D/Fit.D2.RT.Td2Tnd.Main/sbj=S2+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=0+msf=0+ef=1+ec=-1+lf=0+eb=10+td=Ser+smr=NaN+um=2+us=2.mat'
            '../Data_2D/Fit.D2.RT.Td2Tnd.Main/sbj=S3+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=0+msf=0+ef=1+ec=-1+lf=0+eb=10+td=Par+smr=NaN+um=2+us=2.mat'
            '../Data_2D/Fit.D2.RT.Td2Tnd.Main/sbj=S3+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=0+msf=0+ef=1+ec=-1+lf=0+eb=10+td=Ser+smr=NaN+um=2+us=2.mat'
            };
    else
        assert(isempty(S0_file.dfr, []));
        files = {
            '../Data_2D/Fit.D2.RT.Td2Tnd.Main/sbj=S1+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=t+msf=0+ef=1+ec=-1+lf=0+eb=10+td=Par+smr=NaN+um=2+us=2.mat'
            '../Data_2D/Fit.D2.RT.Td2Tnd.Main/sbj=S1+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=t+msf=0+ef=1+ec=-1+lf=0+eb=10+td=Ser+smr=NaN+um=2+us=2.mat'
            '../Data_2D/Fit.D2.RT.Td2Tnd.Main/sbj=S2+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=t+msf=0+ef=1+ec=-1+lf=0+eb=10+td=Par+smr=NaN+um=2+us=2.mat'
            '../Data_2D/Fit.D2.RT.Td2Tnd.Main/sbj=S2+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=t+msf=0+ef=1+ec=-1+lf=0+eb=10+td=Ser+smr=NaN+um=2+us=2.mat'
            '../Data_2D/Fit.D2.RT.Td2Tnd.Main/sbj=S3+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=t+msf=0+ef=1+ec=-1+lf=0+eb=10+td=Par+smr=NaN+um=2+us=2.mat'
            '../Data_2D/Fit.D2.RT.Td2Tnd.Main/sbj=S3+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=t+msf=0+ef=1+ec=-1+lf=0+eb=10+td=Ser+smr=NaN+um=2+us=2.mat'
            };
    end
end

n_subj = numel(subjs);
n_model = 2;
files = flip(reshape(files, [n_model, n_subj])', 2);

%%
Ls = cell(n_subj, n_model);
Ws = cell(n_subj, n_model);
for i_subj = 1:n_subj
    for i_model = 1:n_model
        file1 = files{i_subj, i_model};
        fprintf('Loading %s\n', file1);
        L1 = load(file1);
        L1.Fl.res2W;
        
        Ls{i_subj, i_model} = L1;
        Ws{i_subj, i_model} = L1.Fl.W;
    end
end

%% Average across subjects
datas0 = cell(1, n_subj);
preds0 = cell(n_model, n_subj);
datas = cell(1, n_subj);
preds = cell(n_model, n_subj);
excls = cell(1,n_subj);

% n_dim = Data.Consts.n_dim;
% preds_oversampled = cell(n_dim, n_subj);
% oversample_factor = 10;

for i_subj = 1:n_subj
    for i_model = 1:n_model
        W = Ws{i_subj,i_model};
        
        excl = W.cond_ch_to_exclude;
        data0 = W.Data.RT_data_pdf;
        pred0 = W.Data.RT_pred_pdf;
        
        to_excl = repmat(permute(excl, [5, 1, 2, 3, 4]), ...
            [size(data0, 1), 1, 1, 1, 1]);
        data = data0;
        data(to_excl) = 0;
        pred = pred0;
        pred(to_excl) = 0;
        
%         data = bsxfun(@times, data0, permute(excl, [5, 1, 2, 3, 4]));
%         pred = bsxfun(@times, pred0, permute(excl, [5, 1, 2, 3, 4]));
        
        excls{i_subj} = excl;
        
        datas{i_subj} = data;
        preds{i_model,i_subj} = pred;
        
        datas0{i_subj} = data0;
        preds0{i_model,i_subj} = pred0;
    end
end

%%
data = sum(cat(6, datas{:}), 6);
pred_ser = mean(cat(6, preds{1,:}), 6);
pred_par = mean(cat(6, preds{2,:}), 6);
pred = cat(6, pred_ser, pred_par);

% W1 = W.deep_copy;
% W1.Data.subj = 'S0';
% W1.Data.RT_data_pdf_ = data;
% W1.Data.RT_pred_pdf_ = pred;
% W1.Data.loaded = true;

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

W = Ws{1};
% conds = W.Data.conds;

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
    
    if isequal(S0_file.dfr, 1:4)
        group = [1, 1, 2, 3];
        conds = repmat({[-1, -0.5, -0.25, 0, 0.25, 0.5, 1]}, [1, 2]);
    elseif isequal(S0_file.dfr, 1:6)
        group = [1, 1, 2, 2, 3, 3];
        conds = repmat({[-1, -0.5, -0.25, -0.125, -0.0625, 0, 0.0625, 0.125, 0.25, 0.5, 1]}, [1, 2]);
    else
        group = [1, 1, 2, 2, 3];
        conds = repmat({[-1, -0.5, -0.25, -0.125, 0, 0.125, 0.25, 0.5, 1]}, ...
            [1, 2]);
    end

    h_pred1 = Pl.plot_p(pred1, 'src', 'data', ...
        'condsDim', conds, ...
        'groupAxis', {[], group}, ...
        C{:});
%     h_pred = Pl.plot_p(pred1, 'src', 'pred', C{:});
    hold on;
    set(h_pred1, 'LineStyle', '-', 'LineWidth', 1);

    pred1 = pred(:,:,:,:,:,2);
%     pred1 = Pl.group_p(pred(:,:,:,:,:,model_to_plot), dim_series, ...
%         'group', [1, 2, 2, 3, 3, 3, 2, 2, 1]);
    
    h_pred2 = Pl.plot_p(pred1, 'src', 'data', ...
        'condsDim', conds, ...
        'groupAxis', {[], group}, ...
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
        'condsDim', conds, ...
        'groupAxis', {[], group}, ...
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
        'condsDim', conds, ...
        'groupAxis', {[], group}, C{:});
    set(h_pred1, 'LineStyle', '-', 'LineWidth', 1);
    hold on;

    pred1 = pred(:,:,:,:,:,2);
    Pl = DtbPlot.PlotRt2D;    
    h_pred2 = Pl.plot_p(pred1, 'src', 'pred', ...
        'condsDim', conds, ...
        'groupAxis', {[], group}, C{:});
    set(h_pred2, 'LineStyle', '--', 'LineWidth', 1);
    hold on;

    Pl = DtbPlot.PlotRt2D;
    h_data = Pl.plot_p(data, 'src', 'data', ...
        'condsDim', conds, ...
        'groupAxis', {[], group}, C{:});
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

pth = '../Data_2D/Fit.CompareModels.main_plot_sh_all_subj';
S_file = varargin2S({
    'plt', 'ch_rt'
    }, S0_file);
S2s = bml.str.Serializer;
nam = S2s.convert(S_file);
file = fullfile(pth, nam);
savefigs(file, 'size', [280, 200]);

%% RT-RT
% W = Fit.D2.Bounded.Main;
% C = varargin2C({
%     'dim_on_x', 2
%     'group_y', {4:5, 1:3}
%     });
% h = W.plot_rt_vs_rt_unit(pred_ser, 'style', 'pred', C{:});
% hold on;
% set(h{1}, 'LineStyle', '-');
% 
% h = W.plot_rt_vs_rt_unit(pred_par, 'style', 'pred', C{:});
% hold on;
% set(h{1}, 'LineStyle', '--');
% 
% W.plot_rt_vs_rt_unit(data, 'style', 'data', C{:});
% hold off;

yfun = 'mean';
line_styles = {'-', '--'};
n_data = 5;
f_colormap = @(n_series) {
    flipud(bml.plot.colormaps.winter2(n_series))
    flipud(bml.plot.colormaps.cool2(n_series))
    };
% colormaps = {'winter', 'cool'};
% colormap1 = cool(5);
% colormaps = {cool(5), flipud(winter(5))};
% colormap1 = flipud(winter(5));
% colormaps = {colormap1(:,[2,3,1]), colormap1};

% ylabels = {
%     'Go-RT_{hard M} (s)'
%     'Go-RT_{hard C} (s)'
%     };
% xlabels = {
%     'Go-RT_{easy M} (s)'
%     'Go-RT_{easy C} (s)'
%     };

n_dim = 2;
for dim_on_x = 1:n_dim
    W = L1.Fl.W;
%     W = Fit.D2.Bounded.Main;
    clf;
    
    for i_model = 1:n_model
        pred1 = pred(:,:,:,:,:,i_model);    
        
        if isequal(S0_file.dfr, 1:4)
            group_y = {4, 1:3};
        elseif isequal(S0_file.dfr, 1:6)
            group_y = {5:6, 1:4};
        else
            group_y = {4:5, 1:3};
        end
        
        C = varargin2C({
            'dim_on_x', dim_on_x
            'group_y', group_y
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
        colormaps = f_colormap(n_data);
        
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
            ticks = 0:0.1:5;
        case 2
            ticks = 0:0.1:5;
    end
    set(gca, 'XTick', ticks, 'YTick', ticks);
    
%     bml.plot.beautify_tick;

%     dim_points = n_dim + 1 - dim_on_x;
%     xlabel(xlabels{dim_points});
%     ylabel(ylabels{dim_points});
    
    pth = '../Data_2D/Fit.CompareModels.main_plot_sh_all_subj';
    S_file = varargin2S({
        'plt', 'rt_vs_rt'
        'dmr', dim_on_x
        'yfun', yfun
        }, S0_file);
    S2s = bml.str.Serializer;
    nam = S2s.convert(S_file);
    file = fullfile(pth, nam);

    ax = gca;
    bml.plot.position_subplots(ax, ...
        'margin_top', 0.05, 'margin_right', 0.05, ...
        'margin_left', 0.4, 'margin_bottom', 0.4);
    savefigs(file, 'size', [100, 100]);
end

vars = bml.file.vars_wo_figs;
save('ws_pooled_sh', vars.name);
fprintf('Saved workspace to ws_pooled_sh.mat\n');