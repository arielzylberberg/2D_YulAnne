function main_compare_dtb_all

% clear;
% init_path;

%% 
files = {
    '../Docs/Data_for_paper/Dtb/SerPar-Fit.D2.IrrIxn.Main/sbj={S1,x3}+prd=RT+eor=0+mdl={Ser,Par}+ef=1+ec=-1+fk=0+fsqs=1+fbst=1.mat'
    '../Data_2D/Fit.D2.Inh.MainBatch/sbj={S1,x3}+prd=RT+eor=0+mdl=InhSliceFix+ef=1+ec=-1+fsqs=1+sl0={[0^1,1],x25}.mat'
%     '../Data_2D/Fit.D2.Inh.MainBatch/comp_summary_orig_min_sup15.mat'
%     '../Docs/Data_for_paper/Dtb/Inh-Fit.D2.Inh.MainBatch/sbj={S1,S2,S3}+prd=RT+eor=1+mdl=InhSliceFix+ef=1+ec=-1+fsqs=1+sl0={[0^5,0^5],[1,1],[1,0],[0,1],[1,0^5],[0^5,1],[0,0^5],[0^5,0]}.mat'
    '../Docs/Data_for_paper/Dtb/Trg-Fit.D2.Targetwise.Main/sbj={S1,x3}+prd=RT+eor=0+mdl={Trg}+ef=1+ec=-1+fk=0+fsqs=1+fbst=1.mat'
    '../Docs/Data_for_paper/Dtb/Exv-Fit.D2.Exv.Main/sbj={S1,x3}+prd=RT+eor=0+mdl={Exv}+ef=1+ec=-1+fk=0+fsqs=1+fbst=1.mat'
    '../Data_2D/Fit.D2.Inh.MainBatch/sbj={S1,x3}+prd=RT+eor=0+mdl={InhDrift1}+ef=1+ec=-1+fk=0+fsqs=1+fbst=1.mat'
    '../Data_2D/Fit.D2.Inh.MainBatch/sbj={S1,x3}+prd=RT+eor=0+mdl={InhDrift2}+ef=1+ec=-1+fk=0+fsqs=1+fbst=1.mat'
    '../Data_2D/Fit.D2.Inh.MainBatch/sbj={S1,x3}+prd=RT+eor=0+mdl={InhNoise1}+ef=1+ec=-1+fk=0+fsqs=1+fbst=1.mat'
    '../Data_2D/Fit.D2.Inh.MainBatch/sbj={S1,x3}+prd=RT+eor=0+mdl={InhNoise2}+ef=1+ec=-1+fk=0+fsqs=1+fbst=1.mat'
    };
n_files = numel(files);
Ls = cell(1, n_files);
for ii = 1:n_files
    Ls{ii} = load(files{ii});
end

%% Summarize inh fits
L = Ls{2};
[slprops0, ix] = sortrows(cell2mat2(L.S_comp.slprops0), [1, 2]);
ds = L.ds_cost;
cost = bsxfun(@plus, ds.dcost_best, ds.cost_best);
cost = cost(:, ix);
dcost = bsxfun(@minus, cost, ds.cost_best);
sum_slprops = sum(slprops0, 2);
n_subj = size(cost, 1);
slprops0 = max(slprops0, 0.04);
files = ds.files(:, ix);

%%
fig_tag('scatter_inh');
clf;
sls = unique(slprops0(:,1))';
ax = gobjects(1,n_subj);
for i_subj = 1:n_subj
    ax(i_subj) = subplot(1,n_subj,i_subj);
    z = dcost(i_subj,:) * log(10);
    [~, ix_min] = min(z);    
%     z = tiedrank(z);
%     z_max = prctile(z, 75);
    z_max = 80;
    z = min(z, z_max);
    
    x = slprops0(:,1);
    y = slprops0(:,2);
%     x = 1./x;
%     y = 1./y;
    x_min = x(ix_min);
    y_min = y(ix_min);    
    
%     sls1 = logspace(log10(0.02), log10(1.5));
    sls1 = setdiff(sls, 1);
    sls1 = logspace(log10(sls1(1)), log10(sls1(end)));
    plot(sls1, 1 - sls1, 'k:');
    hold on;
    plot(1 - sls1, sls1, 'k:');
         
    scatter(x, y, 100, z, ...
        'filled'); % , ...
%         'MarkerEdgeColor', 'k');
    hold on;
    plot(x_min, y_min, 'ko', 'MarkerSize', 15);
    hold off;
    
    axis square
%     xlim([0.9, 26]);
%     ylim([0.9, 26]);

    xlim([0.02, 1.5]);
    ylim([0.02, 1.5]);
%     xlim([0.02, 1.05]);
%     ylim([0.02, 1.05]);
    set(gca, 'XTick', sls);
    set(gca, 'YTick', sls);
    set(gca, 'XScale', 'log', 'YScale', 'log', ...
        'XTickLabelRotation', 45);
    colormap(hsv2rev);
    set(gca, 'CLim', [0, z_max]);
    bml.plot.beautify;
    
    %%
    if i_subj == 1
        ylabel('Color accummulation speed');
    else
        set(gca, 'YTickLabel', '');
    end    
    if i_subj == 2
        xlabel('Motion accummulation speed');
        title(sprintf('Subject\n\nS%d', i_subj));
    else
        title(sprintf('S%d', i_subj));
    end
    if i_subj == 3
        %%
        cb = colorbar;
        bml.plot.beautify(cb);
        ylabel(cb, '-{\Delta}log_{10} likelihood');
        yticks = get(cb, 'YTick');
        yticklabels = csprintf('%d', yticks);
        yticklabels{end} = sprintf('> %d', yticks(end));
        set(cb, 'YTickLabel', yticklabels);
    end
    
%     xyz = permute(cat(3, ...
%         [slprops0, zeros(n_model, 1)], ...
%         [slprops0, dcost(i_subj,:)']), [3, 1, 2]);
%     plot3(xyz(:,:,1), xyz(:,:,2), xyz(:,:,3), ...
%         'k-', 'LineWidth', 1);
%     hold on;
%     scatter3(xyz(2,:,1), xyz(2,:,2), ...
%         min(xyz(2,:,3), z_max), ...
%         [], min(xyz(2,:,3), z_max), 'filled', ...
%         'MarkerEdgeColor', 'k');
%     xlim([-0.1, 1.1]);
%     ylim([-0.1, 1.1]);
%     zlim([0, z_max]);
%     set(gca, 'CLim', [0, z_max]);
% %     view(-10, 85);
%     view(0, 90);
%     set(gca, ...
%         'DataAspectRatio', [1, 1, z_max/1], ...
%         'PlotBoxAspectRatio', [1, 1, 1]);
%     colormap(hsv2rev)
end
bml.plot.position_subplots(ax, ...
    'btw_col', 0.02, ...
    'margin_left', 0.1, ...
    'margin_right', 0.18, ...
    'margin_bottom', 0.17, ...
    'margin_top', 0.17);
savefigs('../Data_2D/Fit.main_compare_dtb_all/scatter_inh', ...
    'size', [400, 180]);

%%
mdl_disp_names = {
    'min_sub',          'Serial + Switching'
    'mdl_Ser',          'Serial'
%     'sl0__1_0_',        '1, 0.04'
%     'sl0__0_1_',        '0.04, 1'
%     'sl0__1_1_',        '1, 1'
%     'sl0__0_0_5_',      '(0, 0.5)'
%     'sl0__0_5_0_',      '(0.5, 0)'
%     'sl0__0_5_0_5_',    '(0.5, 0.5)'
%     'sl0__0_5_1_',      '(0.5, 1)'
%     'sl0__1_0_5_',      '(1, 0.5)'
    'min_sup',          'Partial Inhibition'
    'min_sup15',          'Serial + Switching'
    'mdl_Par',          'Parallel'
    'mdl_Exv',          'Exhaustive'
    'mdl_InhNoise1',    'Motion Noise Amp.'
    'mdl_InhNoise2',    'Color Noise Amp.'
    'mdl_InhDrift1',    'Motion Signal Inh.'
    'mdl_InhDrift2',    'Color Signal Inh.'
    'mdl_Trg',          'Targetwise'
    };

mdls = mdl_disp_names(:,1);
mdl_names = mdl_disp_names(:,2);
subjs = Ls{1}.ds_cost.subj;

%% Fill in model costs
ds_cost = dataset;
ds_cost.subj = subjs;
n_subj = numel(subjs);

ds_file = dataset;
ds_file.subj = subjs;

%% Best among super- vs sub-serial
is_sup = sum_slprops > 1;
ix_sup = find(is_sup);
ix_sub = find(sum_slprops < 1);
S_cost = struct;
S_ix = struct;
[S_cost.min_sup, S_ix.min_sup] = min(cost(:, ix_sup), [], 2);
[S_cost.min_sub, S_ix.min_sub] = min(cost(:, ix_sub), [], 2);
S_ix.min_sup = ix_sup(S_ix.min_sup);
S_ix.min_sub = ix_sub(S_ix.min_sub);

for f = {'min_sup', 'min_sub'}
    ds_cost.(f{1}) = S_cost.(f{1});
    for i_subj = 1:n_subj
        ds_file.(f{1}){i_subj,1} = files{i_subj, S_ix.(f{1})(i_subj)};
    end
end

%% Copy model costs
for ii = 1:n_files
    ds1 = Ls{ii}.ds_cost;
    [c, ~, ib] = intersect(mdls, ds1.Properties.VarNames, 'stable');
    for jj = 1:numel(ib)
        c1 = c{jj};
        ds_cost.(c1) = ds1.(c1);
        ds_file.(c1) = ds1.files(:,jj);
    end
end
ds_cost = ds_cost(:, [{'subj'}, mdls(:)']);
ds_file = ds_file(:, [{'subj'}, mdls(:)']);

ds_best = dataset;
ds_best.subj = subjs;
cost_best = nan(n_subj, 1);
i_mdl_best = nan(n_subj, 1);
mdl_best = cell(n_subj, 1);
file_best = cell(n_subj, 1);
for i_subj = 1:n_subj
    [cost_best(i_subj), i_mdl_best(i_subj)] = min(double(ds_cost(i_subj,2:end)));
    mdl_best{i_subj} = mdls{i_mdl_best(i_subj)};
    file_best{i_subj} = ds_file.(mdl_best{i_subj}){i_subj};
end
ds_best.cost_best = cost_best;
ds_best.mdl_best = mdl_best;
ds_best.i_mdl_best = i_mdl_best;
ds_best.file = file_best;

ds_dcost = dataset;
ds_dcost.subj = subjs;
n_mdl = numel(mdls);
for i_mdl = 1:n_mdl
    mdl1 = mdls{i_mdl};
    ds_dcost.(mdl1) = ds_cost.(mdl1) - cost_best;
end

%% Make a bar graph
fig_tag('bar_cost');
clf;
for i_subj = 1:n_subj
    subplotRC(1, n_subj, 1, i_subj);
    dcost1 = double(ds_dcost(i_subj, 2:end)) * log(10);
    barh(dcost1, 'k');
    set(gca, ...
        'YDir', 'reverse');
    
    x_max = max(dcost1(1:(end-1))) * 1.1;
    bml.plot.beautify;
    
    % squiggly line
    hold on;
    
    %%
    bml.plot.squiggly(x_max, n_mdl);
%     for x_shift = 0:0.01:0.1
%         x_squig = 0.95-[-.04, -.01, 0, -.01, -.04, -.05, -.04, -.01]*.7 + x_shift;
%         y_squig = linspace(-0.55, 0.55, 8) + 0.01;
%         plot(x_max * x_squig, n_mdl + y_squig, 'w-', 'LineWidth', 3);
%     end
    
    xlim([0, x_max]);
    ylim([0.5, n_mdl + 0.5]);

    %%
    set(gca, 'XTick', 0:250:1000);
    if i_subj == 1
        set(gca, 'YTickLabel', mdl_names);
    else
        set(gca, 'YTickLabel', '');
    end
    if i_subj == 2
        title(sprintf(['Subject\n\n', subjs{i_subj}]));
        xlabel('-{\Delta}log_{10} likelihood');
    else
        title(subjs{i_subj});
    end
end

%%
ax = bml.plot.subplot_by_pos;
bml.plot.position_subplots(ax, ...
    'margin_top', 0.2, ...
    'margin_left', 0.24, ...
    'margin_right', 0.01, ...
    'margin_bottom', 0.18);
file = '../Data_2D/Fit.main_compare_dtb_all/main_compare_dtb_all';
savefigs(file, ...
    'size', [400, 200]);

%%
save(file);
fprintf('Saved to %s\n', file);

%% Save
export(ds_best, 'File', [file '_ds_best.csv'], 'delimiter', ',');
export(ds_cost, 'File', [file '_ds_cost.csv'], 'delimiter', ',');
export(ds_dcost, 'File', [file '_ds_dcost.csv'], 'delimiter', ',');
fprintf('Exported to %s_ds_best, ds_cost, and _ds_dcost.csv\n', file);

%%
% L = load('../Data_2D/Fit.D2.Inh.MainBatch/comp_summary_orig.mat');