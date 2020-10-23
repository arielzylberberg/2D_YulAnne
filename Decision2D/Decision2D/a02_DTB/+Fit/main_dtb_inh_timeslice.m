% main_dtb_inh_timeslice
% Estimate time slicing coefficient for each stream
% 0 <= pM, pC <= 1
% 0 <= pM + pC <= 2

%% == Fit on cluster
init_path;

S_const0 = varargin2S({
    'UseParallel', 'always' % 'always'|'never'
    'MaxIter', 1e4
    ...
    'parad', 'sh' % 'sh'|'RT'
    'to_excl_outlier_runs', false
    ...
    'model', 'InhSliceFix' % {'InhSliceFree'} % , 'InhSer', 'InhPar'} % {'Ser'}, ... % , 'Par', 'InhFree'}, ...
    'drift_kind', 'IrrSep'
    'bound_kind', 'BetaMeanAsymDec'
    'sigmaSq_kind', 'LinearMinPreDrift' % 'Const'|'LinearMinPreDrift'
    'tnd_kind', 'invgauss'
    'miss_kind', '' % 'Avg'
    'fix_sigmaSq_st', true
    'skip_existing_mat', true
    'skip_existing_fig', true
    });
W0 = Fit.D2.Common.Main;

%%
for to_use_easiest_only = {1} % {0, 1}
    %% Settings - common
    S_hold = varargin2S({
        'subj', Data.Consts.subjs_RT(1) % (1:3)
%         'tr_incl_prct', {[0, 50], [50, 100]}
%         'to_excl_outlier_runs', true
        });
    S_const = varargin2S({
        'to_use_easiest_only', to_use_easiest_only{1}
        'to_use_easiest_only_for_fit', to_use_easiest_only{1}
        'to_use_easiest_only_for_comparison', -to_use_easiest_only{1}
        'force_to_use_easiest_only_for_comparison', 0
        }, S_const0);
    
    slprops0 = {
%         [0.5, 1]
        ...
%         [0, 0.06]
%         [0.06, 0]
%         [0.06, 0.06]
%         [0.06, 0.1]
%         [0.1, 0.06]
        ...
%         [1, 0]
%         [1, 0.1]
%         [1, 0.25]
%         [1, 0.5]
%         [1, 1]
        ...
%         [0.5, 0]
%         [0.5, 0.1]
%         [0.5, 0.25]
%         [0.5, 0.5]
%         [0.5, 1]
%         ...
%         [0.25, 0]
%         [0.25, 0.1]
%         [0.25, 0.25]
%         [0.25, 0.5]
%         [0.25, 1]
%         ...
%         [0.1, 0]
%         [0.1, 0.1]
%         [0.1, 0.25]
%         [0.1, 0.5]
%         [0.1, 1]
%         ...
        [0, 0]
        [0, 0.1]
        [0, 0.25]
        [0, 0.5]
        [0, 1]
        ...
        [0.1, 1]
        [0.25, 1]
        [1, 0.1]
        [1, 0.25]
        [0.1, 0.25]
        ...
        [0.25, 0.1]
        [0.1, 0.5]
        [0.5, 0.1]
        [0.25, 0.5]
        [0.5, 0.25]
        ...
        [0, 0]
        [0, 0.1]
        [0.1, 0]
        [0.1, 0.1]
        [0, 0.25]
        ...
        [0.25, 0]
        [0.25, 0.25]
        [1, 1]
        [1, 0]
        [0, 1]
        ...
        [0.5, 0.5]
        [1, 0.5]
        [0.5, 1]
        [0, 0.5]
        [0.5, 0]
        };
    
    S_comp = varargin2S({
        'slprops0', slprops0
        });
    S_all = varargin2S(varargin2S(S_hold, S_comp), S_const);
    
    C = varargin2C(S_all);

    %% Run on cluster
    W0 = Fit.D2.Common.Main;

    assert(ischar(S_all.parad));
    if strcmp(S_all.parad, 'RT')
        W0.batch_fit_RT_Inh_BetaCdf_Const_Ixn(C{:});
    elseif strcmp(S_all.parad, 'sh')
        W0.batch_fit_sh(C{:});
    end
    
    %% Compare
%     S2s = bml.str.Serializer;
%     Fit.compare_dtb_validation_general(S_comp, S_hold, S_const, ...
%         'replace_file_field', varargin2S({
%             'sl0', S2s.convert(slprops0)
% %             'sl0', S2s.convert({
% %                     slprops0{1}, sprintf('x%d', numel(slprops0))})
%             }));
end

%% == Simulate data from fit
C0 = varargin2C({
    'subj', 'S1'
    }, S_const0);
S_batch = W0.get_S_batch_RT(W0.get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn( ...
    C0{:}));
[Ss, n] = factorizeS(S_batch);

models_data = {'orig'};
n_model_data = numel(models_data);

% {model_fit, subj}
models_fit = {
%     [0.1, 1], [1, 0.1], [0.1, 1]
    [0.5, 1], [1, 0.5], [0.5, 1]
%     [0, 0], [0.1, 0.1], [0, 0.25]
%     [0, 0.5], [0.5, 0], [0, 0.5]
%     [0.5, 0], [0, 0.5], [0.5, 1]
%     [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]
    };
n_model_fit = size(models_fit, 1);

%%
for dif = {1}
%     for subj_seed = {
%              1     4
%              1    10
%              1    15
%              1    19
%              1    20
% 
%              2    10
%              2    15
%              2    18
%              2    19
%              2    20
% 
%              3    10
%              3    14
%              3    15
%              3    18
%              3    19
%              3    20            
%             }'
%         [i_subj, seed] = deal(subj_seed{:});
    for i_subj = 1 % 1:3 % 1:2 % [1, 2, 3]
        for seed = [6, 7, 14] % 1:20
            for i_model_data = 1:n_model_data
                model_data = models_data{i_model_data};
                desc = sprintf('md%s_ef%d', model_data, dif{1});

                if strcmp(model_data, 'orig')
                    model_src = [0, 0.5];
                    src = 'data';
                else
                    error('Not implemented for InhSliceFix yet!');
                    model_src = model_data;
                    src = 'pred';
                end
                
                subj_name = Fit.simulate_dtb([], ... W, ...
                    'desc', desc, ...
                    'seed', seed, ...
                    'src', src, ...
                    ... If model_data='orig' and it is already simulated
                    'subj', Data.Consts.subjs_RT{i_subj}, ...
                    'get_name_only', true);

                %% Fit simulated data
%                 for i_model_fit = 1:n_model_fit
%                     model_fit = models_fit{i_model_fit, i_subj};
% 
%                     S = Ss(1);
%                     S.subj = subj_name;
%                     S = varargin2S({
%                         'model', 'InhSliceFix'
%                         'slprops0', model_fit
%                         'to_use_easiest_only', dif{1}
%                         'to_use_easiest_only_for_fit', dif{1}
%                         'to_use_easiest_only_for_comparison', -dif{1}
%                         }, S);
%                     W0.batch_Ss(S);
%                 end
                
                %% Compare models
                S_hold = varargin2S({
                    'subj', {subj_name}
                    });
                S_comp = varargin2S({
                    'slprops0', models_fit(:, i_subj)'
                    });
                S_const = varargin2S({
                    'to_use_easiest_only', dif{1}
                    'to_use_easiest_only_for_fit', dif{1}
                    'to_use_easiest_only_for_comparison', -dif{1}
                    }, S_const0);
                S2s = bml.str.Serializer;
                Fit.compare_dtb_validation_general( ...
                    S_comp, S_hold, S_const, ...
                    varargin2S({
                        'replace_file_field', varargin2S({
                            'sl0', S2s.convert(models_fit(:,i_subj))
%                             'sl0', sprintf('x%d', n_model_fit)
                        })
                    }));
            end
        end
    end
end

% %% Compare fits to simulation
% subjs = Data.Consts.subjs_RT;
% n_subj = numel(subjs);
% 
% models_data = {'orig'}; % {'Ser', 'Par'}; % 
% n_model_data = numel(models_data);
% 
% models_fit = {'Ser', 'Par'};
% n_model_fit = numel(models_fit);
% 
% for seed = 1:20
%     for dif = {0, 1}
%         subj_data = cell(1, n_model_data * n_subj);
%         n_subj_data = 0;
% % for seed = 2
%         for i_model_data = 1:n_model_data
%             for i_subj = 1:n_subj
%                 model_data = models_data{i_model_data};
%                 desc = sprintf('md%s_ef%d', model_data, dif{1});
%                 subj_name = sprintf('%s_%s_seed%d', ...
%                     subjs{i_subj}, desc, seed);
% 
%                 n_subj_data = n_subj_data + 1;
%                 subj_data{n_subj_data} = subj_name;
%             end
%         end
% % end    
%         S_comp = varargin2S({
%             'model', models_fit
%             }, S_comp0);
%         S_hold = varargin2S({
%             'subj', subj_data(:)'
%             }, S_hold0);
%         S_const = varargin2S({
%             'to_use_easiest_only', dif{1}
%             'to_use_easiest_only_for_fit', dif{1}
%             'to_use_easiest_only_for_comparison', -dif{1}
%             }, S_const0);
% 
%         Fit.compare_dtb_validation_general(S_comp, S_hold, S_const);
%     end
% end

%% Summarize comparison across seeds
comp_kind = 'orig_min_sup15';
switch comp_kind
    % See also Fit.main_dtb_RT
    case 'orig'
        % for orig by subj
        tmp = [
                'sbj={S%3$d_mdorig_ef%1$d_seed%2$d,x1}+prd=RT+eor=0+' ...
                'mdl=InhSliceFix+ef=%1$d+ec=-%1$d+fsqs=1+sl0=x3.mat'];
        model_name = '';
    case 'orig_min_sup15'
        tmp = [
                'sbj={S%3$d_mdorig_ef%1$d_seed%2$d,x1}+prd=RT+eor=0+' ...
                'mdl=InhSliceFix+ef=%1$d+ec=-%1$d+fsqs=1+sl0=%4$s.mat'];
        model_name = 'min_sub15';
end
pth = '../Data_2D/Fit.D2.Inh.MainBatch';
f_file = @(varargin) fullfile(pth, ...
    sprintf(tmp, ...
        varargin{:}));
    
difs = {1}; % {0, 1}
n_dif = numel(difs);
seeds = 1:20;
n_seed = numel(seeds);
files = cell(n_seed, n_dif);
n_subj = numel(Data.Consts.subjs_RT);
ds = dataset;
fields_to_keep = {'subj', 'cost_best', 'i_best', 'best', 'dcost_best', ...
                    'files'};
% Ls = cell(n_seed, n_dif);
for i_dif = 1:n_dif
    for i_subj = 1:n_subj
        for i_seed = 1:n_seed
            dif1 = difs{i_dif};
            seed1 = seeds(i_seed);
            txt_slprops0 = S2s.convert(models_fit(:,i_subj));
            file1 = f_file(dif1, seed1, i_subj, txt_slprops0);
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
                ds1.models = cellfun( ...
                    @(v) bml.str.alphanumeric_name(S2s.convert(v)), ...
                    L1.S_comp.slprops0, ...
                    'UniformOutput', false);
                
                n_row = length(ds1);
                ds1.seed = zeros(n_row, 1) + seed1;
                ds1.ef = zeros(n_row, 1) + dif1;
                ds1.subj0 = cellfun(@(s) s(1:2), ds1.subj, ...
                    'UniformOutput', false);
                if ~isempty(model_name)
                    ds1.models = {model_name};
                end
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
switch comp_kind
    case 'orig_min_sup15'
        file = fullfile('../Data_2D/Fit.D2.Inh.MainBatch', ...
            sprintf('comp_summary_%s', comp_kind));
        ds_cost = ds;
        save(file, 'ds_cost');
        export(ds, 'file', [file, '.csv'], 'Delimiter', ',');
        fprintf('Saved to %s.mat and csv\n', file);
end

%% Load comparison made in main_dtb_RT
L = load('../Data_2D/Fit.D2.IrrIxn.Main/comp_summary_orig.mat');
fields_to_keep_ds1 = setdiff([fields_to_keep, {'seed', 'ef', 'subj0'}], ...
    {'subj'}, 'stable');
ds1 = L.ds(:, fields_to_keep_ds1);
ds1.cost = bsxfun(@plus, ds1.cost_best, ds1.dcost_best);
ds1.models = repmat({'Ser', 'Par'}, [length(ds1), 1]);

%% Merge ds0 and ds1
fields_to_keep_final = [fields_to_keep_ds1, {'cost', 'models'}];
fields_to_concatenate = {'cost', 'models', 'files'};

ds = ds0;
for ii = 1:length(ds0)
    ds01 = ds0(ii,:);
    ds11 = bml.ds.find(ds1, varargin2S({
        'subj0', ds01.subj0{1}
        'seed', ds01.seed(1)
        'ef', ds01.ef(1)
        }), fields_to_keep_final);
    assert(length(ds11) == 1);
    
    for f = fields_to_concatenate
        v = [ds11.(f{1})(1,:), ds01.(f{1})(1,:)];
        len_v = length(v);
        ds.(f{1})(ii,1:len_v) = v;
    end
    
    [ds.cost_best(ii), ds.i_best(ii)] = min(ds.cost(ii,:));
    ds.dcost_best(ii, 1:len_v) = bsxfun(@minus, ...
        ds.cost(ii,:), ds.cost_best(ii,:));
    ds.best{ii} = ds.models{ii,ds.i_best(ii)};
end

varnames0 = ds.Properties.VarNames;
varnames_first = { ...
    'subj0', 'seed', 'ef', 'best', 'cost_best', 'i_best', 'cost', ...
    'dcost_best', 'models', 'files'};
ds = ds(:, [varnames_first, setdiff(varnames0, varnames_first, 'stable')]);
disp(ds);

%%
file = fullfile('../Data_2D/Fit.D2.Inh.MainBatch', ...
    sprintf('comp_summary_%s', comp_kind));
save(file, 'ds');
export(ds, 'file', [file, '.csv'], 'Delimiter', ',');
fprintf('Saved to %s.mat and csv\n', file);
