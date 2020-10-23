classdef MainFull < GLM.Ch.Main
    % Fit.Rt.MainFull
    %
    % 2017 YK wrote the initial version.
properties
    mdl_full % The full model
end
methods
    function W = MainFull(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function [Ss, files, csv_file, pow_file] = batch(W0, varargin)
        S_batch = W0.get_S_batch(varargin{:});
        [Ss, n] = factorizeS(S_batch);
        files = cell(n, 1);
        for ii = 1:n
            S = Ss(ii);
            C = varargin2C(S);
            
            W = feval(class(W0), C{:});
            if isempty(W.Data.ds)
                continue;
            end
            W0.W_now = W;
            W.main;
            files{ii} = [W.get_file, '.mat'];
        end
        files = files(~cellfun(@isempty, files));
        csv_file = W0.summarize(files);
        pow_file = struct;
%         pow_file = W0.calc_power(files, csv_file);
    end
    function batch_short(W0, varargin)
        C = varargin2C(varargin, {
            'subj', Data.Consts.subjs_short, ...
            'parad', {'sh'}
            });
        W0.batch(C{:});
    end
    function S_batch = get_S_batch_short(W, varargin)
        S_batch = W.get_S_batch(varargin2S(varargin, {
            'subj', Data.Consts.subjs_short, ...
            'parad', {'sh'}
            }));
    end
    function S_batch = get_S_batch(~, varargin)
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT_incl_monk
            'parad', {'RT'}
            'dim_rel_W', {1, 2}
            'n_dim_task', {1, 2}
            });
    end
    function fit(W)
        y = W.get_ch;
        X = W.get_X;
        
        if isempty(y) || isempty(X)
            W.mdl_full = [];
        else
            W.mdl_full = fitglm(X, y, ...
                'Distribution', 'binomial', ...
                'VarNames', W.VarNames, ...
                'ResponseVar', 'Choice');
        end
    end
    function savefigs(W)
        file = [W.get_file({'tbl', 'mdl'}), '.txt'];
        if exist(file, 'file')
            delete(file);
        end
        mkdir2(fileparts(file));
        diary(file);
        t = evalc('W.mdl_full');
        t = strrep(strrep(t, '<strong>', ''), '</strong>', '');
        disp(t);
        diary('off');
    end
    function X = get_X(W)
        cond = W.get_cond_2D;
        dim_rel = W.get_dim_rel_W;
        dim_irr = W.get_dim_irr_W;
        
        rel = cond(:, dim_rel);
        irr = cond(:, dim_irr);
        irr = irr ./ max(abs(irr));

        accu = double(W.Data.ds.accu);
        accu(irr == 0, dim_irr) = nan;
        accu(rel == 0, dim_rel) = nan;
        
        accu_irr = accu(:, dim_irr);
        
        n_prev = 1;
        n_dim = 2;
        accu1 = zeros(size(accu, 1) - n_prev, n_dim);
        for i_prev = 1:n_prev
            accu1 = accu1 + accu(i_prev:(end - n_prev - 1 + i_prev), :);
        end
        accu1 = accu1 ./ n_prev;
        accu_prev = [nan(n_prev, n_dim); accu1];
        
        % Combine accu_prev across dims
%         accu_prev1 = accu_prev;
%         clear accu_prev
%         accu_prev = mean(accu_prev1, 2);
        
        X = [rel, irr, rel .* irr, ...
             abs(irr), rel .* abs(irr), ...
             rel .* accu_prev];
%         X = [rel, irr, rel .* irr, ...
%              1 - abs(irr), rel .* (1 - abs(irr))];
         
        if W.n_dim_task == 2
            X = [X, accu_irr, rel .* accu_irr];
        end
%          
        X = standardize(X);
        
%         X = bsxfun(@minus, X, nanmean(X));
    end    
    function v = get_VarNames(W)
        if W.n_dim_task == 2
%             v = {'r', 'i', 'rxi', 'ai', 'rxai', 'Choice'};
            v = {'r', 'i', 'rxi', 'ai', 'rxai', 'ci', 'rxci', ...
                'rxpar', 'rxpai', ...
                ... 'rxpa' ... 
                'Choice'};
        else
            v = {'r', 'i', 'rxi', 'ai', 'rxai', 'Choice'};
%             v = {'r', 'i', 'rxi', 'di', 'rxdi', 'Choice'};
        end
    end
    function v = get_VarNames_long(W)
        if W.n_dim_task == 2
            v = {'rel', 'irr', 'rel_x_irr', 'abs_irr', 'rel_x_abs_irr', ...
                'accu_irr', 'rel_x_accu_irr', ...
                'rel_x_prev_accu_rel', 'rel_x_prev_accu_irr', ...
                ... 'rel_x_prev_accu', ... 
                'Choice'};        
%             v = {'rel', 'irr', 'rel_x_irr', 'abs_irr', 'rel_x_abs_irr', ...
%                 'Choice'};        
%             v = {'rel', 'irr', 'rel_x_irr', 'dif_irr', 'rel_x_dif_irr', ...
%                 'Choice'};        
        else
            v = {'rel', 'irr', 'rel_x_irr', 'dif_irr', 'rel_x_dif_irr', ...
                'Choice'};        
        end
    end
end
%% Summarize
methods
    function csv_file = summarize(W0, mat_files)
        if nargin < 2
            mat_files = W0.get_mat_files;
        end
        n = numel(mat_files);
        ds = dataset;
        for ii = n:-1:1
            mat_file = mat_files{ii};
            
            row = W0.summarize_unit(mat_file);
            ds = ds_set(ds, ii, row);
        end
        
        S_file0 = W0.S_file_;
        W0.S_file_ = struct;
        file = W0.get_file({'tbl', 'summary'});
        W0.S_file_ = S_file0;
        
        csv_file = [file, '.csv'];
        export(ds, 'File', csv_file, 'Delimiter', ',');
    end
    function row = summarize_unit(W0, mat_file)
        %%
        L = load(mat_file);
        row = struct;
        
        row = copyFields(row, L.W, {
            'subj'
            'parad'
            'task'
            'dif_rel_incl'
            'dif_irr_incl'
            'accu_irr_incl'
            'n_dim_task'
            'dim_rel_W'
            });
        
        row.subj = strrep_cell( ...
            row.subj, Data.Consts.subjs_strrep);
        
        % end-1 because the last one is the response variable.
        var_names = [{'Intercept'}, W0.VarNames_long(1:(end-1))];
        
        n_var = size(L.W.mdl_full.Coefficients, 1);
        for i_var = 1:n_var
            str = sprintf('%1.2f +- %1.2f (p=%1.3g)', ...
                L.W.mdl_full.Coefficients.Estimate(i_var), ...
                L.W.mdl_full.Coefficients.SE(i_var), ...
                L.W.mdl_full.Coefficients.pValue(i_var));
            
            var_name = var_names{i_var};
            row.(var_name) = str;
        end
    end
end
%% Power calculation
methods
    function pow_file = calc_power(W0, mat_files, csv_file, varargin)
        if nargin < 2 || isempty(mat_files)
            mat_files = W0.get_mat_files;
        end
        if nargin < 3 || isempty(csv_file)
            csv_file = W0.get_csv_file;
        end
        
        %%
        n = numel(mat_files);
        ds = dataset('File', csv_file, 'Delimiter', ',');
        
        power_thres_all = cell(1, n);
        for ii = 1:n
            power_thres_all{ii} = ...
                W0.calc_power_unit(mat_files{ii}, varargin{:});
        end
        for ii = 1:n
            po = power_thres_all{ii};
            for col = fieldnames(po)'
                ds.(col{1}){ii, 1} = po.(col{1});
            end
        end
        
        %%
        S_file0 = W0.S_file_;
        W0.S_file_ = struct;
        pow_file = [W0.get_file({'tbl', 'pow'}), '.csv'];
        W0.S_file_ = S_file0;
        
        export(ds, 'File', pow_file, 'Delimiter', ',');
        fprintf('Power calculation saved to %s\n', pow_file);
    end
    function [power_thres_str, power_thres] = ...
            calc_power_unit(W0, mat_file, varargin)
        % [po_str, po] = calc_power_unit(W0, mat_file, varargin)
        
        S = W0.get_S_calc_power(varargin{:});
        
        L = load(mat_file);
        mdl = L.W.mdl_full;
        
        if S.skip_existing && isfield(L, 'power_thres')
            power_thres = L.power_thres;
            fprintf('Skip calculating existing power_thres for %s\n', ...
                mat_file);
        else       
            power_thres = struct;
        end
        power_thres_str = struct;
        
        n_coefs = numel(S.coefs);
        for ii = n_coefs:-1:1
            coef = S.coefs{ii};
            if ~ismember(coef, mdl.CoefficientNames)
                continue;
            end

            % Old format. Backward compatibility.
            col1 = sprintf('%s_%d_beta_%1.0f', ...
                    coef, 1, S.beta*100);
                
            % New format. Separates power_thres from power_thres_str.
            col0 = sprintf('%s_beta_%1.0f', ...
                    coef, S.beta*100);
            if isfield(power_thres, col0) && S.skip_existing
                v_res = power_thres.(v_res);
                
            elseif isfield(power_thres, col1) && S.skip_existing
                v_res = zeros(1, 2);
                for jj = 2:-1:1
                    col = sprintf('%s_%d_beta_%1.0f', ...
                        coef, jj, S.beta*100);
                    v_res(jj) = str2double(power_thres.(col));
                end                
            else
                v_res = bml.stat.logistic_effect_size( ...
                    mdl, coef, ...
                    'beta', S.beta, ...
                    'to_plot', true);
            end
            
            estimate_rel = table2array(mdl.Coefficients('r', 'Estimate'));
            for jj = 1:2
                col = sprintf('%s_%d_beta_%1.0f', ...
                    coef, jj, S.beta*100);
                
                ix_coef = find(strcmp(mdl.CoefficientNames, coef));
                power_thres_str.(col) = sprintf('%1.2f (%1.1f%% of rel)', v_res(jj), ...
                    v_res(jj) / abs(estimate_rel) * 100);
            end
        end
        
        power_thres_str.n_tr = sprintf('%1.0f', ...
            size(L.W.mdl_full.Variables, 1));
        power_thres_str.df = sprintf('%1.0f', ...
            L.W.mdl_full.DFE);
        
        L.power_thres = power_thres;
        L.power_thres_str = power_thres_str;
        save(mat_file, '-struct', 'L');
        fprintf('Saved power calculation results to %s\n', mat_file);
    end
    function S = get_S_calc_power(~, varargin)
        S = varargin2S(varargin, {
            'coefs', {'rxai'} % {'rxdi'} % {'rxai', 'rxci'}
            'beta', 0.8
            'skip_existing', true
            });    
    end
    function dat_files = repeat_data(W0, dat_file0, varargin)
        S = varargin2S(varargin, {
            'n_rep', [10, 100]
            });
        
        if nargin < 2
            dat_file0 = '../Data_2D/sTr/RT_FR.mat';
        end
        [pth, nam0] = fileparts(dat_file0);
        
        L0 = load(dat_file0);
        
        n = numel(S.n_rep);
        dat_files = cell(n, 1);
        for ii = 1:n
            n_rep1 = S.n_rep(ii);
            dat_file1 = fullfile(pth, [nam0, sprintf('x%d', n_rep1)]);
            
            L = L0;
            L.dat = repmat(L.dat, [n_rep1, 1]);
            save(dat_file1, '-struct', 'L');
            fprintf('Saved to %s\n', dat_file1);
            dat_files{ii} = dat_file1;
        end
    end
end
%% plot
methods
    function plot_ch(W, varargin)
        S = varargin2S(varargin, {
            'dat', []
            'dim_rel', 1
            'kind', 'by_accu_irr'
            });
        if isempty(S.dat)
            dat = W.Data.ds;
        else
            dat = S.dat;
        end
        ch = [dat.subjM, dat.subjC];
        cond = [dat.condM, dat.condC];
        n_dim = 2;
        conds = cell(1, n_dim);
        
        for dim = 1:n_dim
            [conds{dim},~,adcond(:,dim)] = unique(abs(cond(:,dim)));
        end
        n_adcond = max(adcond, [], 1);
        accu = sign(ch - 1.5) == sign(cond);
        
        %%
        switch S.kind
            case 'by_accu_irr'                
                dim_rel = S.dim_rel;
                dim_irr = n_dim + 1 - dim_rel;
                
                specs = {'--', '-'};
                colors = hsv2rev(n_adcond(dim_irr));
                
                adcond_irr = adcond(:,dim_irr);
                accu_irr = accu(:, dim_irr);
                for i_adcond = 1:n_adcond(dim_irr)
                    for accu = 0:1
                        incl = (accu_irr == accu) ...
                            & (i_adcond == adcond_irr) ...
                            & (cond(:,dim_rel) ~= 0);
                        [cond_incl, ~, adcond_incl] = ...
                            unique(cond(incl, dim_rel));
                        ch_incl = ch(incl, dim_rel);
                        
                        conds_incl = unique(cond_incl);
                        p_ch_cond = accumarray(adcond_incl, ...
                            ch_incl, [], @mean);
                            
%                         if accu == 0
%                             args = varargin2C({
%                                 'MarkerFaceColor', 
                        plot(conds_incl, p_ch_cond, ...
                            'o', ...
                            'Color', colors(i_adcond, :), ...
                            'LineStyle', specs{accu + 1});
                        
%                         h = bml.stat.plot_binned_ch(cond_incl, ch_incl == 2, ...
%                             'plot_args', {
%                                 'Color', colors(i_adcond, :)
%                                 'LineStyle', specs{accu + 1}
%                                 });
                        
%                         h = bml.stat.glmplot(cond_incl, ch_incl == 2, ...
%                             'binomial');
%                         set(h, ...
%                             'Color', colors(i_adcond, :), ...
%                             'LineStyle', specs{accu + 1});                            
                        hold on;
                    end
                end
                
            otherwise
                error('Unknown kind=%s\n', S.kind);
        end
    end
end
%% mat files
methods
    function mat_files = get_mat_files(~)
        mat_files = {
%             '../Data_2D/GLM.Ch.MainFull/sbj=DX+prd=RT+tsk=A+dtk=2+dmr=1+trm=201+eor=t+nv=8+n_sim=100.mat'
%             '../Data_2D/GLM.Ch.MainFull/sbj=DX+prd=RT+tsk=A+dtk=2+dmr=2+trm=201+eor=t+nv=8+n_sim=100.mat'
%             '../Data_2D/GLM.Ch.MainFull/sbj=DX+prd=RT+tsk=H+dtk=1+dmr=1+trm=201+eor=t+nv=6+n_sim=100.mat'
%             '../Data_2D/GLM.Ch.MainFull/sbj=DX+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+nv=6+n_sim=100.mat'
%             '../Data_2D/GLM.Ch.MainFull/sbj=MA+prd=RT+tsk=A+dtk=2+dmr=1+trm=201+eor=t+nv=8+n_sim=100.mat'
%             '../Data_2D/GLM.Ch.MainFull/sbj=MA+prd=RT+tsk=A+dtk=2+dmr=2+trm=201+eor=t+nv=8+n_sim=100.mat'
%             '../Data_2D/GLM.Ch.MainFull/sbj=MA+prd=RT+tsk=H+dtk=1+dmr=1+trm=201+eor=t+nv=6+n_sim=100.mat'
%             '../Data_2D/GLM.Ch.MainFull/sbj=MA+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+nv=6+n_sim=100.mat'
%             '../Data_2D/GLM.Ch.MainFull/sbj=VL+prd=RT+tsk=A+dtk=2+dmr=1+trm=201+eor=t+nv=8+n_sim=100.mat'
%             '../Data_2D/GLM.Ch.MainFull/sbj=VL+prd=RT+tsk=A+dtk=2+dmr=2+trm=201+eor=t+nv=8+n_sim=100.mat'
%             '../Data_2D/GLM.Ch.MainFull/sbj=VL+prd=RT+tsk=H+dtk=1+dmr=1+trm=201+eor=t+nv=6+n_sim=100.mat'
%             '../Data_2D/GLM.Ch.MainFull/sbj=VL+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+nv=6+n_sim=100.mat'
            '../Data_2D/GLM.Ch.MainFull/sbj=FR+prd=RT+tsk=A+dtk=2+dmr=1+trm=201+eor=t+nv=8+n_sim=100.mat'
%             '../Data_2D/GLM.Ch.MainFull/sbj=FR+prd=RT+tsk=A+dtk=2+dmr=2+trm=201+eor=t+nv=8+n_sim=100.mat'
            };
    end
    function csv_file = get_csv_file(~)
        csv_file = '../Data_2D/GLM.Ch.MainFull/tbl=summary.csv';
    end
        
end
end