% main_compare_model_RT
% global to_use_parallel

clear;
init_path;
W0 = Fit.D2.Common.Main;

%% == Common settings
S_hold0 = varargin2S({
    'subj', Data.Consts.subjs_RT(4) % :3)
    'tr_incl_prct', {[0, 100]} % 25]}
    ... 'tr_incl_prct', {[0, 50], [50, 100]}
    'sigmaSq_kind', 'LinearMinPreDrift' %'Const'|'LinearMinPreDrift'
    'to_excl_outlier_runs', false
    });
S_comp0 = varargin2S({
    'model', {'Par'} % , 'Par'} %  {'Ser', 'Par', 'Trg', 'Exv', 'InhDrift1', 'InhDrift2', 'InhNoise1', 'InhNoise2'} % 'Ser', 'Par'} % {'min_sub', 'Ser', 'min_sup125', 'min_sup15', 'Par'} % 'InhDrift2', 'InhNoise1', 'InhNoise2'} % 'InhNoise1', 'InhNoise2'} % {'Ser', 'Par', 'Trg', 'Exv'} % {'Ser', 'Par', 'InhSer', 'InhPar', 'InhSlice', 'InhSliceFix', 'InhSliceFree', 'InhFree'}
    });
S_const0 = varargin2S({
    'parad', 'RT' % 'RT'|'sh'
    'drift_kind', 'IrrSep'
    'bound_kind', 'BetaMeanAsymDec' % 'BetaMeanAsymDec'|'BetaMeanAsym'|'BMA2'|'CosBasis'
    'tnd_kind', 'invgauss'
    'kbratio_kind', 'n'
    'disper_kind', 'std'
    'miss_kind', '' % 'Avg'|''
    'fix_sigmaSq_st', true % false % 
    'fix_kappa', false % true
%     'fix_fano', false
    ...
    'fix_bias_st', true
    'fix_irr_ixn', false % True: use Bounded.Main; False: use IrrIxn.Main
    'to_fix_bias_irr', true
    ...
    'dif_rel_incl', 1:5 % Try excluding easiest
    'dif_irr_incl', 1:5 % Try excluding easiest
    });
S0 = varargin2S(S_hold0, varargin2S(S_comp0, S_const0));
C0 = varargin2C(S0);
difs = {1}; % {[1, 2]}; % , [1, 2]}; % {0, 1} % Use [1,2] for FR since higher coherence is involved
dif = difs(1);

%% Comomn settings used in Proj.FixKappa.FixKappa
% init_path;
% W0 = Fit.D2.Common.Main;
% S0 = varargin2S({
%     'model', {'Ser', 'Par'} % , 'Par'} % , 'Par'} % 'InhSliceFree'} % , 'InhSer', 'InhPar'} % {'Ser'}, ... % , 'Par', 'InhFree'}, ...
%     'drift_kind', 'IrrSep'
%     'bound_kind', 'BetaMeanAsym' % 'BetaMeanAsym'|'BetaMeanAsymDec'|'BMA2'
%     'tnd_kind', 'invgauss'
%     'kbratio_kind', 'n'
%     ... 'disper_kind', 'std'
%     'miss_kind', '' % 'Avg'|''
%     'fix_sigmaSq_st', false % true
%     ... 'fix_kappa', true
%     ...
%     'to_excl_outlier_runs', true
%     ...
% %     'fix_bias_st', false
% %     'fix_irr_ixn', false
% %     'to_fix_bias_irr', false
%     'sigmaSq_kind', 'Const' % 'Const'|'LinearMinPreDrift'
%     });

%% Fit on cluster 
for dif = difs
    S_const = varargin2S({
        'to_use_easiest_only', dif{1}
        'to_use_easiest_only_for_fit', dif{1}
        'to_use_easiest_only_for_comparison', -dif{1}
        ...
        'skip_existing_mat', true 
        'skip_existing_fig', true
        ...
        'MaxIter', 1e4 % 1 % 
        'UseParallel', 'always' % 'never' % 
        ...
        'th0', varargin2S({
            'Dtb__Dtb1__Bound__b', 0.8
            'Dtb__Dtb2__Bound__b', 0.8
            });
%         ...
%         'dt', 1/25
        }, S_const0);
    C = varargin2C(S_const, S0);
    
    %% Fit
    W0.batch_fit_RT_Inh_BetaCdf_Const_Ixn(C{:});
    
    %% Plot
%     W0.batch_plot_RT_Inh_BetaCdf_Const_Ixn(C{:});

    %% Compare cost
    Fit.compare_dtb_validation_general(S_comp0, S_hold0, S_const);
end

%% == Commenting out below for running on cluster

% %% Batch plot / local testing
% for dif = difs
%     C = varargin2C({
%         'subj', Data.Consts.subjs_RT(:)'
%         'MaxIter', 1
%         ...
%         'to_use_easiest_only', dif{1}
%         'to_use_easiest_only_for_fit', dif{1}
%         'to_use_easiest_only_for_comparison', -dif{1}
%         ...
%         'skip_existing_mat', false % true %
%         'skip_existing_fig', false
%         'UseParallel', 'never'
%         }, S0);
%     
% %     % fit
% %     W0.batch_fit_RT_Inh_BetaCdf_Const_Ixn(C{:});
%     
% %     % plot
% %     W0.batch_plot_RT_Inh_BetaCdf_Const_Ixn(C{:});
% 
%     % create
%     S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn(C{:});
%     Ss = factorizeS(S);
% 
%     W = W0.create_RT(Ss(1));    
% end
% 
% %% == Compare cost
% % for dif = difs
% %     C = varargin2C({
% %         'subj', Data.Consts.subjs_RT
% %         ...
% %         'to_use_easiest_only', dif{1}
% %         'to_use_easiest_only_for_fit', dif{1}
% %         'to_use_easiest_only_for_comparison', -dif{1}
% %         }, C0);
% % 
% %     [ds_cost, file_ds] = Fit.compare_dtb_validation(C{:});
% % end
% 
% %% == Simulate data from fit
% models_data = {'orig'};
% n_model_data = numel(models_data);
% 
% models_fit = {'Par'}; % 'min_sup125'}; % {'Ser', 'Par'}; % , 'min_sub'}; % , 'min_sup'};
% n_model_fit = numel(models_fit);
% 
% %%
% for dif = difs
% %     for i_subj = 3 % 1:3
% %         subj = subjs{i_subj};
% 
%     for subj_seed = {
% %             'S1', 12
% %             'S1', 16
%             'S2', 12
%             }'
%         [subj, seed] = deal(subj_seed{:});
% 
% %     for seed = 1:20
%         C1 = varargin2C({
%             'subj', subj
%             }, C0);
% 
%         S_batch = W0.get_S_batch_RT(W0.get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn( ...
%             C1{:}));
%         S_batch = bml.struct.rmfield(S_batch, {'model', 'to_use_easiest_only'});
%         [Ss, n] = factorizeS(S_batch);
% 
%         for ii = 1:n
%             for i_model_data = 1:n_model_data
%                 model_data = models_data{i_model_data};
%                 desc = sprintf('md%s_ef%d', model_data, dif{1});
% 
%                 if strcmp(model_data, 'orig')
%                     model_src = 'Ser';
%                     src = 'data';
%                 else
%                     model_src = model_data;
%                     src = 'pred';
%                 end
%                 
%                 S = Ss(ii);            
%                 C = varargin2C({
%                     'model', model_src
%                     'skip_existing_mat', false
%                     'skip_existing_fig', false
%                     'to_use_easiest_only', dif{1}
%                     'to_use_easiest_only_for_fit', dif{1}
%                     'to_use_easiest_only_for_comparison', -dif{1}
%                     }, S);
%                 
% %                 if (seed == 12 && strcmp(S.subj, 'S1')) || ...
% %                         (seed == 12 && strcmp(S.subj, 'S2')) || ...
% %                         (seed == 16 && strcmp(S.subj, 'S1'))
% %                 else
% %                     continue;
% %                 end
%                 
%                 W = W0.create(C{:});
% 
%                 subj_name = Fit.simulate_dtb(W, ...
%                     'desc', desc, ...
%                     'seed', seed, ...
%                     'src', src, ...
%                     'get_name_only', true);
% 
%                 %% Fit simulated data
%                 for i_model_fit = 1:n_model_fit
%                     model_fit = models_fit{i_model_fit};
% 
% %                     if (seed == 12 && strcmp(S.subj, 'S1') && ismember(model_fit, {'Ser', 'Par'})) || ...
% %                             (seed == 12 && strcmp(S.subj, 'S2') && strcmp(model_fit, 'Par')) || ...
% %                             (seed == 16 && strcmp(S.subj, 'S1') && ismember(model_fit, {'Ser', 'Par'}))
% %                     else
% %                         continue;
% %                     end                    
%                     
%                     S = Ss(ii);
%                     S.subj = subj_name;
%                     S = varargin2S({
%                         'model', model_fit
%                         'to_use_easiest_only', dif{1}
%                         'to_use_easiest_only_for_fit', dif{1}
%                         'to_use_easiest_only_for_comparison', -dif{1}
% %                         'MaxIter', 1
%                         }, S);
% 
%                     %%
%                     W0.batch_Ss(S);
%                 end
%             end
%         end
% %     end
%     end
% end
% 
% %% Compare fits to simulation
% subjs = Data.Consts.subjs_RT;
% n_subj = numel(subjs);
% 
% models_data = {'orig'}; % {'Ser', 'Par'}; % 
% n_model_data = numel(models_data);
% 
% models_fit = {'min_sup125', 'min_sup15', 'min_sub', 'Ser', 'Par', 'min_sup'}; % {'min_sup15'}; % 
% n_model_fit = numel(models_fit);
% 
% for seed = 5:20
%     for dif = {1} % {0, 1}
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
% 
% %% Summarize comparison across seeds
% comp_kind = 'orig';
% switch comp_kind
%     case 'orig'
%         % for orig
%         tmp = [
%                 'sbj={S1_mdorig_ef%1$d_seed%2$d,x3}+prd=RT+eor=0+' ...
%                 'mdl={min_sub,Ser,Par}+ef=%1$d+ec=-%1$d+fk=0+fsqs=1+fbst=1.mat'];
% %         tmp = [
% %                 'sbj={S1_mdorig_ef%1$d_seed%2$d,x3}+prd=RT+eor=1+' ...
% %                 'mdl={Ser,Par}+ef=%1$d+ec=-%1$d+fk=1+fsqs=0+fbst=1.mat'];
%     case 'SerPar'
%         % for Ser vs Par
%         tmp = [
%                 'sbj={S1_mdSer_ef%1$d_seed%2$d,x6}+prd=RT+eor=1+' ...
%                 'mdl={Ser,Par}+ef=%1$d+ec=-%1$d+fk=1+fsqs=0+fbst=1.mat'];
% end
% pth = '../Data/Fit.D2.Inh.MainBatch';
% % pth = '../Data/Fit.D2.IrrIxn.Main';
% f_file = @(ef, seed) fullfile(pth, ...
%     sprintf(tmp, ...
%         ef, seed));
%     
% difs = {1}; % {0, 1};
% n_dif = numel(difs);
% seeds = 1:20;
% n_seed = numel(seeds);
% files = cell(n_seed, n_dif);
% ds = dataset;
% % Ls = cell(n_seed, n_dif);
% for i_dif = 1:n_dif
%     for i_seed = 1:n_seed
%         dif1 = difs{i_dif};
%         seed1 = seeds(i_seed);
%         file1 = f_file(dif1, seed1);
%         files{i_seed, i_dif} = file1;
%         if exist(file1, 'file')
%             L1 = load(file1);
% %             Ls{i_seed, i_dif} = load(file1);
%             fprintf('Loaded %s\n', file1);
%             
%             ds1 = L1.ds_cost;
%             n_row = length(ds1);
%             ds1.seed = zeros(n_row, 1) + seed1;
%             ds1.ef = zeros(n_row, 1) + dif1;
%             ds = [ds; ds1]; %#ok<AGROW>
%         else
%             warning('File absent: %s\n', file1);
%         end
%     end
% end
% 
% %%
% ds.model_data = cellfun(@(s) s(6:8), ds.subj, 'UniformOutput', false);
% ds.model_best = cellfun(@(s) s(5:7), ds.best, 'UniformOutput', false);
% ds.correct_model = strcmp(ds.model_data, ds.model_best);
% ds.dcost_sgn = sign(diff(ds.dcost_best, 1, 2));
% ds.subj0 = cellfun(@(s) s(1:2), ds.subj, 'UniformOutput', false);
% 
% [subjs, ~, ix_subj] = unique(ds.subj0);
% n_subj = numel(subjs);
% seed = ds.seed;
% [~,~,ix_dif] = unique(ds.ef);
% switch comp_kind
%     case 'orig'
%         models_data = {'ori'};
%     case 'SerPar'
%         models_data = {'Ser', 'Par'};
% end
% models_pred = {'min', 'Ser', 'Par'}; % {'Ser', 'Par'};
% ix_model_data = strcmpfinds(ds.model_data, models_data);
% ix_model_pred = strcmpfinds(ds.model_best, models_pred);
% ix_data = [seed, ix_subj, ix_model_data, ix_dif];
% ix_pred = [seed, ix_subj, ix_model_pred, ix_dif];
% incl = ~any(isnan(ix_data), 2);
% 
% mdl_count = accumarray(ix_pred(incl, :), 1);
% sum_count = permute(sum(mdl_count), [2, 3, 4, 1]);
% 
% mdl_correct = accumarray(ix_data(incl, :), ds.correct_model(incl));
% mdl_wrong = accumarray(ix_data(incl, :), ~ds.correct_model(incl), ...
%     [], @sum, nan);
% 
% sum_correct = permute(sum(mdl_correct), [2, 3, 4, 1]);
% sum_wrong = permute(sum(mdl_wrong), [2, 3, 4, 1]);
% 
% n_total = sum_correct + sum_wrong;
% p_correct = sum_correct ./ n_total;
% 
% txt_file = fullfile(pth, sprintf('comp_summary_%s.txt', comp_kind));
% if exist(txt_file, 'file')
%     delete(txt_file);
% end
% diary(txt_file);
% disp('sum_count (subj, model_pred, ef)');
% disp(sum_count);
% 
% disp('p_correct (subj, model_data, ef)');
% disp(p_correct);
% 
% disp('n_total (subj, model_data, ef)');
% disp(n_total);
% diary('off');
% fprintf('Saved to %s\n', txt_file);
% 
% mat_file = fullfile(pth, sprintf('comp_summary_%s.mat', comp_kind));
% save(mat_file, ...
%     'ds', 'models_data', 'models_pred', ...
%     'mdl_count', 'sum_count', ...
%     'mdl_correct', 'sum_correct', ...
%     'mdl_wrong', 'sum_wrong', ...
%     'n_total', 'p_correct');
% fprintf('Saved to %s\n', mat_file);
% 
% %% == Within-Condition Cross-validation (legacy)
% % init_path;
% % file0 = '../Data/Fit.D2.IrrIxn.Main/sbj=DX+prd=RT+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl=Par+dft=S+bnd=A+ssq=C+tnd=i+ntnd=4+msf=0+ef=0+ec=-0+lf=0+td=Par+fsqs=0+fbst=1.mat';
% % [pth, nam] = fileparts(file0);
% % cvn = 2;
% % file1 = fullfile(pth, [nam ...
% %     sprintf('+cvn=%d+cvi=0.mat', cvn)]);
% % copyfile(file0, file1);
% % 
% % p_mdl_correct = mdl_correct ./ (mdl_correct + mdl_wrong);
% % 
% % %%
% % rng(0);
% % Cv = CrossvalFl;
% % Cv.fit_opts = varargin2C({
% %     'UseParallel', 'never'
% %     'MaxIter', 1
% %     }, Cv.fit_opts);
% % Cv.file_orig = file1;
% % Cv.files_res = cell(1, cvn);
% % 
% % for cvi = 1:cvn
% %     Cv.files_res{cvi} = fullfile(pth, [nam ...
% %         sprintf('+cvn=%d+cvi=%d.mat', cvn, cvi)]);
% % end
% % 
% % L = load(file1);
% % Fl = L.Fl;
% % Fl.res2W;
% % 
% % % %%
% % % Fl.W.to_exclude_bins_wo_trials = 5;
% % 
% % %%
% % Cv.fit_Fl(Fl, 'n_set', 2);
% 
% 
% %% Legacy
% % % %% Fit on Cluster - ef=1,2
% % % init_path;
% % % W0 = Fit.D2.Common.Main;
% % % for dif = {0} % 1, 1:2} % {1:4, 1:3}
% % %     C = varargin2C({
% % %         'subj', Data.Consts.subjs_RT
% % %         'model', {'Par', 'Ser'} % , 'InhSliceFree', 'InhSlice'} % , 'InhSer', 'InhPar'} % {'Ser'}, ... % , 'Par', 'InhFree'}, ...
% % %         'drift_kind', 'IrrSep'
% % %         'bound_kind', 'BetaMeanAsym'
% % %         'tnd_kind', 'invgauss'
% % %         'miss_kind', '' % 'Avg'|''
% % %         'fix_sigmaSq_st', false
% % %         'to_excl_outlier_runs', true
% % %         'skip_existing_mat', false % true %
% % %         'skip_existing_fig', false
% % %         'UseParallel', 'always' % 'never' % 
% % %         ... 'MaxIter', 0
% % %         'to_use_easiest_only', dif{1}
% % %         'to_use_easiest_only_for_fit', dif{1}
% % %         'to_use_easiest_only_for_comparison', -dif{1}
% % %         });
% % %     W0.batch_fit_RT_Inh_BetaCdf_Const_Ixn(C{:});
% % % end
% % % 
% % %% Local testing
% % % init_path;
% % % W0 = Fit.D2.Common.Main;
% % % dif = {1:2};
% % % 
% % % C = varargin2C({
% % %     'subj', Data.Consts.subjs_RT(1)
% % %     'model', {'InhSliceFix'} % , 'InhSer', 'InhPar'} % {'Ser'}, ... % , 'Par', 'InhFree'}, ...
% % %     ...
% % %     'drift_kind', 'IrrSep'
% % %     'bound_kind', 'BetaMeanAsym'
% % %     'tnd_kind', 'invgauss'
% % %     'miss_kind', '' % 'Avg'|''
% % %     'fix_sigmaSq_st', true
% % %     'to_excl_outlier_runs', true
% % %     'skip_existing_mat', false % true %
% % %     'skip_existing_fig', false
% % %     'UseParallel', 'always' % 'never' % 
% % %     ... 'MaxIter', 0
% % %     'to_use_easiest_only', dif{1}
% % %     'to_use_easiest_only_for_fit', dif{1}
% % %     'to_use_easiest_only_for_comparison', -dif{1}
% % %     ...
% % %     'slprops0', {[1, 0], [1, 1], [0, 1], ...
% % %                  [1, 0.5], [0.5, 0.5], [0.5, 1], ...
% % %                  [0, 0.5], [0.5, 0]}
% % %     ...
% % %     'UseParallel', 'never' % 
% % %     'MaxIter', 0
% % %     });
% % % W0.batch_fit_RT_Inh_BetaCdf_Const_Ixn(C{:});
% % 
% % %%
% % S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn(C{:});
% % Ss = factorizeS(S);
% % 
% % W = W0.create_RT(Ss(1));
% % 
% % %%
% % % WPar = W0.create_RT(Ss(2));
% % 
% % %%
% % % WPar.th.Dtb__Dtb1__Drift__bias = WPar.th0.Dtb__Dtb1__Drift__bias + 0.1;
% % % WPar.pred;
% % % WPar.th.Dtb__Dtb1__Drift__bias = WPar.th0.Dtb__Dtb1__Drift__bias;
% % W.pred;
% % W.plot_plotfuns
% % 
% % %%
% % W.main;
% % 
% % % %% Fit on Cluster
% % % to_use_parallel = true;
% % % 
% % % init_path;
% % % W0 = Fit.D2.Common.Main;
% % % for dif = {1, 2, 3, 0} % 1, 1:2} % {1:4, 1:3}
% % %     C = varargin2C({
% % %         'subj', Data.Consts.subjs_RT
% % %         'model', {'Par', 'Ser'}
% % %         'drift_kind', 'IrrSep'
% % %         'bound_kind', 'BetaMeanAsym'
% % %         'tnd_kind', 'invgauss'
% % %         'miss_kind', '' % 'Avg'|''
% % %         'fix_sigmaSq_st', false
% % %         'to_excl_outlier_runs', true
% % %         'skip_existing_mat', false % true %
% % %         'skip_existing_fig', false
% % %         'UseParallel', 'always' % 'never' % 
% % %         ... 'MaxIter', 0
% % %         'to_use_easiest_only', dif{1}
% % %         'to_use_easiest_only_for_fit', dif{1}
% % %         'to_use_easiest_only_for_comparison', -dif{1}
% % %         });
% % %     W0.batch_fit_RT_Inh_BetaCdf_Const_Ixn(C{:});
% % % end
% % 
% % %% == Compare cost
% % clear;
% % init_path;
% % 
% % for dif = {0, 1}
% %     C = varargin2C({
% %         'subj', Data.Consts.subjs_RT
% %         'model', {'Ser', 'Par', 'InhSlice', 'InhSliceFree'} % {'Ser', 'Par'} % , 'InhSer', 'InhPar'} % {'Ser'}, ... % , 'Par', 'InhFree'}, ...
% %         'drift_kind', 'IrrSep'
% %         'bound_kind', 'BetaMeanAsym' % 'BMA2' % 'Dec'
% %         'tnd_kind', 'invgauss'
% %         'fix_sigmaSq_st', false %  false
% %         'to_excl_outlier_runs', true
% %         'skip_existing_mat', false % true %
% %         'skip_existing_fig', false
% %         'UseParallel', 'always' % 'never' % 
% %         ... 'MaxIter', 0
% %         'to_use_easiest_only_for_fit', dif{1}
% %         'to_use_easiest_only_for_comparison', -dif{1}
% %         });
% % 
% %     [ds_cost, file_ds] = Fit.compare_dtb_validation(C{:});
% % end