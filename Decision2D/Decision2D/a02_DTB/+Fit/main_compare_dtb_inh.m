%%
% Produced in Fit.main_dtb_RT, %% Compare fits to simulation
tmp = [
    'sbj={S%3$d_mdorig_ef%1$d_seed%2$d,x3}+prd=RT+eor=0' ...
    '+mdl={min_sup125,min_sup15,min_sub,Ser,Par,min_sup}' ...
    '+ef=%1$d+ec=-%1$d+fk=0+fsqs=1+fbst=1.mat'];

% tmp = [
%     'sbj={S%3$d_mdorig_ef%1$d_seed%2$d,x3}+prd=RT+eor=0' ...
%     '+mdl={min_sub,Ser,min_sup,Par}' ...
%     '+ef=%1$d+ec=-%1$d+fk=0+fsqs=1+fbst=1.mat'];

pth = '../Data_2D/Fit.D2.Inh.MainBatch';
models0 = {'min_sup125', 'min_sup15', 'min_sub', 'Ser', 'Par', 'min_sup'};
models = {'min_sub', 'Ser', 'min_sup125', 'min_sup15', 'Par'};
yticklabels = {
        'Serial + Switching'
        'Serial'
        '{\Sigma}Speed = 1.25'
        '{\Sigma}Speed = 1.5'
        'Parallel'
        };
[~,ix_models] = ismember(models, models0);

f_file = @(varargin) fullfile(pth, ...
    sprintf(tmp, ...
        varargin{:}));

difs = {1};
n_dif = numel(difs);
seeds = 1:20;
n_seed = numel(seeds);
files = cell(n_seed, n_dif);
n_subj = numel(Data.Consts.subjs_RT);
subjs = Data.Consts.subjs_RT;
ds = dataset;
fields_to_keep = {'subj', 'cost_best', 'i_best', 'best', 'dcost_best', ...
                    'files'};
% Ls = cell(n_seed, n_dif);
for i_dif = 1:n_dif
    for i_subj = 1 % :n_subj
        for i_seed = 1:n_seed
            dif1 = difs{i_dif};
            seed1 = seeds(i_seed);
            file1 = f_file(dif1, seed1, i_subj);
            files{i_seed, i_dif, i_subj} = file1;
            if exist(file1, 'file')
                L1 = load(file1);
    %             Ls{i_seed, i_dif} = load(file1);
                fprintf('Loaded %s\n', file1);

                ds1 = L1.ds_cost;
                ds1 = ds1(:, ...
                    fields_to_keep);
%                 ds1.slprops0 = L1.S_comp.slprops0;
                S2s = bml.str.Serializer;
%                 ds1.models = cellfun( ...
%                     @(v) bml.str.alphanumeric_name(S2s.convert(v)), ...
%                     L1.S_comp.slprops0, ...
%                     'UniformOutput', false);
                
                n_row = length(ds1);
                ds1.seed = zeros(n_row, 1) + seed1;
                ds1.ef = zeros(n_row, 1) + dif1;
                ds1.subj0 = cellfun(@(s) s(1:2), ds1.subj, ...
                    'UniformOutput', false);
                ds = [ds; ds1]; %#ok<AGROW>
            else
                warning('File absent: %s\n', file1);
            end
        end
    end
end
ds.cost = bsxfun(@plus, ds.cost_best, ds.dcost_best);
ds.subj = [];
disp(ds);
ds0 = ds;

%%
fig_tag('cost_bootstrap');
clf;
ax = subplotRCs(1, n_subj);
costs = cell(1, n_subj);
dcosts = cell(1, n_subj);
pvals = cell(1, n_subj);

ds_txt = dataset;
markers = {'*', '**', '***'};
p_thres = [0.05, 0.005, 0.001];

for i_subj = 1:n_subj
    subj = subjs{i_subj};
    incl = strcmp(ds.subj0, subj);
    
    cost = ds.cost(incl,:) * log(10);
    cost = cost(:,ix_models);
    costs{i_subj} = cost;
    
    dcost = bsxfun(@minus, cost, cost(:,1));
    mean_cost = nanmean(dcost);
    sem_cost = nansem(dcost);
    [~, pval1] = ttest(dcost);

    dcosts{i_subj} = dcost;
    pvals{i_subj} = pval1;
    
    for i_model = 1:n_model
        model = models{i_model};
        ds_txt.(model){i_subj,1} = sprintf('%g +- %g (p=%g)', ...
            mean_cost(i_model), sem_cost(i_model), pval1(i_model));
    end
    
%     min_mean_cost = nanmin(mean_cost);
%     dcost_mean = mean_cost - min_mean_cost;
%     disp(dcost_mean);
    
    ax1 = ax(i_subj);
    axes(ax1);
    n_model = size(cost, 2);
    barh(1:n_model, mean_cost, 'FaceColor', 0.7 + zeros(1,3));
    hold on;
    bml.plot.errorbar_wo_tick2(mean_cost, 1:n_model, sem_cost, ...
        [], [], [], {
        'Marker', 'none'
        'LineStyle', 'none'
        }, {
        'LineStyle', '-'
        'LineWidth', 2
        });
    ylim([0.5, n_model + 0.5]);
    x_lim = [nanmin([0, mean_cost - sem_cost]), ...
          nanmax(mean_cost + sem_cost) * 1.25];
    xlim(x_lim);
    
    dx_lim = diff(x_lim);
    for i_model = 2:n_model
        s = sign(mean_cost(i_model));
        pos = mean_cost(i_model) + (sem_cost(i_model) + dx_lim / 20) * s;
        ix_pval = find(pval1(i_model) < p_thres, 1, 'last');
        txt1 = markers{ix_pval};
        
        if s > 0
            align = 'Left';
        else
            align = 'Right';
        end
        text(pos, i_model-0.1, txt1, 'FontSize', 15, ...
            'HorizontalAlignment', align);
    end
    
    set(ax1, 'YDir', 'reverse');
    if i_subj == 1
        set(ax1, 'YTickLabel', yticklabels);
    else
        set(get(ax1, 'YAxis'), 'Visible', 'off');
    end
    bml.plot.beautify;
    
    if i_subj == 2
        txt = sprintf('Subject\n\n%s', subj);
        xlabel('-{\Delta}log_{10} likelihood');
    else
        txt = subj;
    end
    title(ax1, txt);
end
bml.plot.position_subplots(ax, ...
    'margin_top', 0.2, ...
    'margin_bottom', 0.18, ...
    'margin_right', 0.05, ...
    'margin_left', 0.3);

%%
% for i_subj = 1:n_subj
%     [row, col] = find(isnan(costs{i_subj}));
%     incl = col < 4;
%     row = row(incl);
%     
%     disp([i_subj + zeros(size(row, 1),1), row]);
% end

disp(ds_txt);

file = '../Data_2D/Fit.main_compare_dtb_inh/cost_bootstrap_inh';
export(ds_txt, 'file', [file '.csv'], 'Delimiter', ',');
fprintf('Saved to %s.csv\n', file);

savefigs(file, 'size', [300, 200]);
save(file, 'costs', 'dcosts', 'pvals', 'models');
fprintf('Saved results to %s.mat\n', file);
