classdef Main < GLM.Ch.CommonWorkspace
    % GLM.Ch.Main
    %
    % 2016 YK wrote the initial version.
properties
    res = struct;
    mdl
    info
    mdls
    
    W_now = []; 
    
    n_sim = 1e2; % 1e4;
%     crossval_method = 'HoldOut';
    p_holdout = 0.5;
end
properties (Dependent)
    VarNames
    VarNames_long
    n_var
end
%% Batch
methods
    function Ss = batch_short(W0, batch_args)
        if ~exist('batch_args', 'var'), batch_args = struct; end
        Ss = W0.batch(W0.get_S_batch_short(batch_args));
    end
    function [Ss, files] = batch(W0, batch_args)
        if ~exist('batch_args', 'var'), batch_args = struct; end
        S_batch = W0.get_S_batch(batch_args);
        [Ss, n] = factorizeS(S_batch);
        files = cell(n, 1);
        for ii = 1:n
            S = Ss(ii);
            C = varargin2C(S);
            
            W = feval(class(W0), C{:});
            W.main;
            files{ii} = [W.get_file, '.mat'];
        end
    end
    function S_batch = get_S_batch_short(W, batch_args)
        if ~exist('batch_args', 'var'), batch_args = struct; end
        S_batch = W.get_S_batch(varargin2S(batch_args, {
            'subj', Data.Consts.subjs_short, ...
            'parad', {'sh'}
            }));
    end
    function S_batch = get_S_batch(~, batch_args)
        if ~exist('batch_args', 'var'), batch_args = struct; end
        S_batch = varargin2S(batch_args, {
            'subj', Data.Consts.subjs_RT
            'parad', {'RT'}
            'dim_rel_W', {1, 2}
            'n_dim_task', {1, 2}
            });
    end
    function main(W)
%         if ~exist([W.get_file '.mat'], 'file') ...
%                 || ~W.skip_existing_mat
            W.fit;
            W.save_mat;
%         else
%             warning('Found existing fit! Loading..');
%             W = W.load_mat;
%         end
        W.savefigs;
    end
end
%% Initialization
methods
    function W = Main(varargin)
        W.dt = W.max_t * 4;
        
        W.file_fields = [W.file_fields
            {
                'n_sim', ''
                'p_holdout', ''
            }];
        W.file_mult = [W.file_mult
            {
                'p_holdout', 100
            }];
        
        if nargin > 0
            W.init(varargin{:});
        end
    end
end
%% Fit
methods
    function fit(W)
        %%
        y = W.get_ch;
        X = W.get_X;
        
        if isempty(y) || isempty(X)
            W.mdl = [];
            W.info = struct;
            W.mdls = [];
        else
            [W.mdl, W.info, W.mdls] = bml.stat.fitglm_exhaustive(X, y, {
                'Distribution', 'binomial', ...
                'VarNames', W.VarNames, ...
                'ResponseVar', 'Choice'
                }, ...
                'model_criterion', 'crossval', ...
                'crossval_args', {
                    'crossval_method', 'HoldOut', ... % W.crossval_method, ...
                    'crossval_args', {W.p_holdout}, ...
                    'n_sim', W.n_sim
                    }, ...
                'must_include', 1);
        end
    end
    function pred(W)
        % Use W.mdl fitted from W.fit
        
    end
    function X = get_X(W)
        cond = W.get_cond_2D;
        dim_rel = W.get_dim_rel_W;
        dim_irr = W.get_dim_irr_W;
        accu_irr = W.Data.ds.accu(:, dim_irr);
        
        rel = cond(:, dim_rel);
        irr = cond(:, dim_irr);
        
        X = [rel, irr, rel .* irr, ...
             abs(irr), rel .* abs(irr)];
         
        if W.n_dim_task == 2
            X = [X, accu_irr, rel .* accu_irr];
        end
         
%         X = standardize(X);
        
%         X = bsxfun(@minus, X, nanmean(X));
    end
    function v = get.VarNames(W)
        v = W.get_VarNames;
    end
    function v = get_VarNames(W)
        if W.n_dim_task == 2
            v = {'r', 'i', 'rxi', 'ai', 'rxai', 'ci', 'rxci', 'Choice'};
        else
            v = {'r', 'i', 'rxi', 'ai', 'rxai', 'Choice'};
        end
    end
    function v = get.VarNames_long(W)
        v = W.get_VarNames_long;
    end
    function v = get_VarNames_long(W)
        if W.n_dim_task == 2
            v = {'rel', 'irr', 'rel_x_irr', 'abs_irr', 'rel_x_abs_irr', ...
                'accu_irr', 'rel_x_accu_irr', 'Choice'};        
        else
            v = {'rel', 'irr', 'rel_x_irr', 'abs_irr', 'rel_x_abs_irr', ...
                'Choice'};        
        end
    end
end
%% Plot goodness of fit
methods
    function plot_ic(W)
        n_model = length(W.info.ic_all);
        
        errorbar_wo_tick(1:n_model, W.info.ic_all, ...
            W.info.ic_all_se, [], {'k', 'LineStyle', 'none'}, {'k-'});
        
        hold on;
        
        %% Mark
        ix = W.info.ic_min_ix;
        
        y_lim = ylim;
        y_mark = W.info.ic_all(ix) + W.info.ic_all_se(ix) ...
               + (y_lim(2) - y_lim(1)) / 15; % ...
%             max(W.info.ic_all_se(ix) * 1.5, ...
%                 (y_lim(2) - y_lim(1)) / 10);
        
        plot(ix, y_mark, 'k*');
        hold off;
        
        ylabel('Mean Negative Log Likelihood');

        W.xlabel_ic;
        bml.plot.beautify;
        grid on;
        
        title(W.get_title({'fig', 'plot_ic'}));
    end
    function boxplot_ic(W)
        ic_all0 = cell2mat(W.info.ic_all0(:)');        
        boxplot(ic_all0);
        
        ylabel('Negative Log Likelihood');

        W.xlabel_ic;
        bml.plot.beautify;

        title(W.get_title({'fig', 'boxplot_ic'}));
    end
    function ecdf_ic(W, varargin)
        S = varargin2S(varargin, {
            'n_to_plot', 3
            });
        
        [~, ix0] = sort(W.info.ic_all);
        
        for ii = 1:S.n_to_plot
            ix = ix0(ii);
            ecdf(W.info.ic_all0{ix});
            hold on;
        end
        hold off;
        
        model_labels = W.get_model_labels;
        legend(model_labels(ix0(1:S.n_to_plot)), ...
            'Location', 'SouthEast');
        bml.plot.beautify;
        grid on;
        
        title(W.get_title({'fig', 'ecdf_ic'}));
    end
    function h = plot_slice(W)
        h = W.mdl.plotSlice;        
    end
    function xlabel_ic(W)
        n_model = length(W.info.ic_all);
        ticklabels = W.get_model_labels;
        set(gca, 'XTick', 1:n_model, 'XTickLabel', ticklabels, ...
            'XTickLabelRotation', 45);
        
        xlabel('Parameters Included');
    end
    function model_labels = get_model_labels(W)
        n_model = length(W.info.ic_all);
        model_labels = cell(1, n_model);
        for i_model = 1:n_model
            model_labels{i_model} = ...
                str_bridge(' + ', ...
                    W.VarNames{W.info.param_incl_all(i_model, :)});
%                 sprintf('%d', W.info.param_incl_all(i_model, :));
        end
    end
end
%% Save
methods
    function files = savefigs(W)
        figure('Visible', 'off');
        
        files = {};
        for kind = {'plot_ic', 'plot_slice', 'boxplot_ic', 'ecdf_ic'}
            try
                file = W.get_file({'fig', kind{1}});
                mkdir2(fileparts(file));
                
                if exist([file '.fig'], 'file') ...
                        && W.skip_existing_fig
                    
                    fprintf('Skipping existing figure %s\n', ...
                        [file '.fig']);
                    continue;
                end
                
                clf;
                W.(kind{1});
                
                files0 = savefigs(file);

                files = [files(:); files0(:)];
            catch err
                fprintf('===== Error plotting %s\n', kind{1});
                warning(err_msg(err));
            end
        end
    end
    function save_mat(W0)
        file = W0.get_file;
        W = W0.obj2struct; %#ok<NASGU>
        
        mkdir2(fileparts(file));
        save([file '.mat'], 'W', 'W0');
    end
    function W0 = load_mat(W0, file)
        if ~exist('file', 'var')
            file = W0.get_file;
        end
        L = load([file '.mat']);
        if isfield(L, 'W0') && strcmp(class(L.W0), class(W0))
            if nargout >= 1
                W0 = L.W0;
            else
                bml.oop.copyprops(W0, L.W0);
            end
        else
            bml.oop.copyprops(W0, L.W);
        end
    end
    function v = get.n_var(W)
        v = numel(W.VarNames);
    end
    function fs = get_file_fields0(W)
        fs = [
            W.get_file_fields0@GLM.Ch.CommonWorkspace
            {
            'n_var', 'nv'
            }];
    end
end
%% Table
methods
    function ds = save_table_short(W0, batch_args)
        if ~exist('batch_args', 'var'), batch_args = struct; end
        S_batch = W0.get_S_batch_short(batch_args);
        ds = W0.save_table(S_batch);
    end
    function ds = save_table(W0, batch_args)
        if ~exist('batch_args', 'var'), batch_args = struct; end
        S_batch = W0.get_S_batch(batch_args);
        [Ss, n] = factorizeS(S_batch);
        
%         s = cell(n, 1);
        ds_s = dataset;
        ds = dataset;
        files = cell(n, 1);
    
        for row = n:-1:1
            S = Ss(row);
            
            W = eval(class(W0));
            varargin2props(W, S);
            file = W.get_file;
            mkdir2(fileparts(file));
            
            fprintf('Loading %s\n', file);
            W = W.load_mat(file);
            
            [s1, S1] = W.get_row;
            ds = ds_set(ds, row, S1);
            ds_s = ds_set(ds_s, row, s1);
            files{row} = file;
%             s{row} = s1;
        end
        ds.file = files;
        
        %%
        S2s = bml.str.Serializer;
        file_tab = fullfile('Data', class(W0), S2s.convert(S_batch));
        export(ds_s, 'file', [file_tab '.csv'], 'Delimiter', ',');
        fprintf('Saved %s\n', [file_tab '.csv']);
        
        ds_s.file = files;
        xlwrite([file_tab '.xlsx'], ds_s);
        fprintf('Saved %s\n', [file_tab '.xlsx']);
        
%         fid = fopen([file_tab '.csv'], 'w');
%         fprintf(fid, '%s\n', s{:});
%         fclose(fid);
        
        save(file_tab, 'ds');
        fprintf('Saved ds to %s\n', [file_tab '.mat']);
    end
    function [s, S] = get_row(W)
        S.Subj = W.subj(1);
        S.Parad = W.parad;
        S.N_Dim = W.n_dim_task;
        S.Feature = W.Data.dimNames_long{W.dim_rel_W};
        S.df = W.mdl.DFE;
        
        s = S;
        
        cols = strrep_cell(W.mdl.CoefficientNames, {
            '(Intercept)', 'Intercept'
            });
        n_cols = numel(cols);
        
        for ii = 1:n_cols
            col = cols{ii};
            [s.(col), S1] = W.get_cell(W.mdl.Coefficients(ii,:));
            S = copyFields(S, bml.struct.prefix_fields(S1, ['th_' col '_']));
        end
    end
    function [s, S] = get_cell(~, coef)
        S.est = coef.Estimate;
        S.se = coef.SE;
        S.tstat = coef.tStat;
        S.pval = coef.pValue;
        S.incl = true;
        
        s = sprintf('%1.2f (p=%1.1g)%s', ...
            S.est, S.pval, bml.str.pval2marks(S.pval));
    end
end
end