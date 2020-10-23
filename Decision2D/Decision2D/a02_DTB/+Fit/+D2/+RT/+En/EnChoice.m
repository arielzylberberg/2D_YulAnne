classdef EnChoice < Fit.D2.Common.CommonWorkspace & Fit.D2.Common.CommonWorkspace
    % Fit.D2.RT.En.EnChoice
    %
    % 2015 YK wrote the initial version
properties (SetAccess = protected)
    Bases
    Leverage
end
properties
    Leverage_super = 'zEn.EnLeverage'
    Leverage_available = varargin2S({
        'Mean',   @zEn.EnLeverageMean
        'Glm',    @zEn.EnLeverageGlm
        'GlmSep', @zEn.EnLeverageGlmSep
        'Lasso',  @zEn.EnLeverageLasso
        });
end
%% Init
methods
    function W = EnChoice
        W.set_Bases;
        W.set_Leverage;
        W.set_Data;
    end
end
%% Batch
methods
    function batch_demo(W, path_args, args_factor, args_common)
        %%
        if ~exist('path_args', 'var'), path_args = {}; end
        [~, ~, Ss, n_file] = Data.DataLocator.sTr(path_args);
        n_dim = W.Data.get_n_dim;
        
        if ~exist('args_factor', 'var')
            args_factor = {
                'dim_rel_W', 1:n_dim
                'Ixn', {'Rel'}
                'Leverage', {'Lasso'} % {'GlmSep'}
                'dif_incl_rel', {1, 1:2}
                'dif_incl_irr', {1, 2, 3, 4, 5, 2:3, 4:5}
                }; 
        end
        if ~exist('args_common', 'var'), args_common = {}; end
        
        size_figs = varargin2S({
            'b', [400, 600]
            'prop_valid', [400, 600]
            });
        
        args_factor = varargin2S(args_factor);
        [Ss_factor, n_S] = factorizeS(args_factor);
        S_common = varargin2S(args_common);
        
        %%
        for i_file = 1:n_file
            W.Data.loaded = false;
            S_path = Ss(i_file);
            C_path = S2C(S_path);
                
            n_S = numel(Ss_factor);
            for ii = 1:n_S
                S = varargin2S(Ss_factor(ii), S_common);
                S_kind = varargin2S(S, {
                    'tag', ''
                    'dim_rel_W', 1
                    'Ixn', ''
                    'dif_incl_rel', 1
                    'dif_incl_irr', 1
                    });
                C_kind = S2C(S_kind);
                res = W.demo('path_args', C_path, C_kind{:});

                for tag = res.fig_tag(:)'
                    fig_tag(tag{1});
                    S_kind = varargin2S({
                        'tag', tag{1}
                        }, S_kind);
                    C_kind = S2C(S_kind);

                    file = W.get_file_res( ...
                        'path_args', C_path, ...
                        'kind', C_kind);
                    
                    if isfield(size_figs, tag{1})
                        size_fig = size_figs.(tag{1});
                    end
                    savefigs(file, 'size', size_fig);
                end
            end
        end
    end
    function file = get_file_res(W, varargin)
        S = varargin2S(varargin, {
            'path_args', {}
            'kind', {}
            });
        S.path_args = varargin2C(S.path_args);
        [file, ~, S_path] = Data.DataLocator.sTr(S.path_args{:});
        [~, file] = fileparts(file{1});
        
        S.kind = varargin2S({
            'src', file
            }, S.kind);
        
        pth = fullfile('Data', class(W), varargin2dir(S.kind)); % class2dir(class(W))
        file = varargin2name(S.kind);
        
        file = fullfile(pth, file);
    end
end
%% Demo
methods
    function res = demo(W, varargin)
        S = varargin2S(varargin, {
            'path_args', {'subj', 'VL'}
            'dim_rel_W', 1
            't_incl_min', 0
            't_incl_max', 1
            'shift_ix', []
            'width_cycle_sec', 0.1
            'Leverage', 'GlmSepIxn' % 'Lasso' % 'GlmSep'
            'Ixn', 'RelPool' % 'Rel'
            'dif_incl_rel', 1
            'dif_incl_irr', 4
            });
        res = struct;
        res.fig_tag = {};
        
        if ~W.Data.loaded
            W.set_dt(1/75);
            W.Data.set_path(S.path_args);
            W.Data.load_data;
        
            %%
            W.Bases.set_width_per_cycle_sec(S.width_cycle_sec);
            W.Bases.adapt_Data;
        end
        
        %% Exclude anomalous data
        W.Data.set_filt_spec(@(Dat) (Dat.ds0.task == 'A') & ...
            nanmean(abs(cell2mat2(Dat.ds0.mCE)), 2) < 20);
        W.Data.filt_ds;
        
        dim_rel_W = S.dim_rel_W;
        W.set_dim_rel_W(dim_rel_W);
        dim_rel = W.get_dim_rel_W;
        dim_irr = W.get_dim_irr_W;
        
        n_dim = W.Data.get_n_dim;
        
        %% Filter by difficulty
        n_tr = length(W.Data.ds);
        tr_incl = true(n_tr, 1);
        
        adCond = W.Data.get_adCond;
%         aConds = W.Data.get_aConds;

        dif_incl = cell(1, 2);
        dif_incl{dim_rel} = S.dif_incl_rel;
        dif_incl{dim_irr} = S.dif_incl_irr;
        
        for dim = 1:n_dim
            tr_incl = tr_incl & ...
                bsxEq(adCond(:, dim), dif_incl{dim}); % abs(cond(:, dim)), aConds{dim}(dif_incl{dim}));
        end
        
        %%
%         % TODO: accu considering mean En and bias 
%         % - Perhaps inappropriate since this is an RT task
%         %   and subjects did not ignore a feature due to the other feature.
%         accu = W.Data.get_accu;
%         tr_incl = tr_incl & accu(:, dim_irr);
        fprintf('# trial included = %d\n', nnz(tr_incl));
        
        %% Type of Leverage and Ixn
        W.set_Leverage(S.Leverage);
        W.Leverage.set_Ixn(S.Ixn);
        W.Leverage.LocalTime.set_dt(W.Bases.get_wavelength_in_sec / 4);
        
        if strcmp(S.Ixn, 'Rel')
            if isempty(S.shift_ix)
                n_shift_half = floor( ...
                    S.t_incl_max / W.Bases.get_wavelength_in_sec * 4 / 2);
                S.shift_ix = -n_shift_half:n_shift_half;
            end
            W.Leverage.Ixn.set_shift_ix(S.shift_ix);
        end
        
        %% Add ch
        ch = W.Data.get_ch;
        ch_rel = ch(tr_incl, dim_rel) == 2;
        ch_irr = ch(tr_incl, dim_irr) == 2;
        W.Leverage.set_ch(ch_rel);
        
        %% Get X
        ix_t_incl_max = ceil(S.t_incl_max / W.get_dt);
        ix_t_incl_max = ceil(ix_t_incl_max ...
            / W.Bases.get_wavelength_in_bin * 4);
        
        ix_t_incl_min = ceil(S.t_incl_min / W.get_dt);
        ix_t_incl_min = ceil(ix_t_incl_min ...
            / W.Bases.get_wavelength_in_bin * 4);
        ix_t_incl = (ix_t_incl_min:ix_t_incl_max) + 1;
        
        Xs = cell(1, n_dim);
%         Xs{1} = W.Bases.Bases{dim_rel}.wt(tr_incl, :); % Wrong dim
%         Xs{2} = W.Bases.Bases{dim_irr}.wt(tr_incl, :); % Wrong dim

        for ii = 1:n_dim
%             Xs{ii} = W.Bases.Bases{ii}.wt(tr_incl, :);
            Xs{ii} = W.Data.Ens{ii}.get_ts_mat;
            Xs{ii} = Xs{ii}(tr_incl, :);
        end
        
        %% Add basic regressors
        W.Leverage.reset_regressors;
        ix_reg = cell(1, n_dim);
%         for dim = 1:n_dim % W.get_dim_rel_W
% %             W.Leverage.add_regressors(cond(tr_incl, dim));
%             ix_reg{dim} = ...
%                 W.Leverage.add_regressors(Xs{dim}(tr_incl, 1:n_t_incl));
%         end
        ix_reg{1} = ...
            W.Leverage.add_regressors(Xs{dim_rel}(:, ix_t_incl), 'rel');
        
        mean_X_irr = nanmean(Xs{dim_irr}, 2);
        res_mean_irr = glmwrap(mean_X_irr, ch_irr, 'binomial');
        bias = -res_mean_irr.b(1) / res_mean_irr.b(2);
        
        res_mean_irr_sanity = glmwrap(mean_X_irr - bias, ch_irr, 'binomial');
        disp(res_mean_irr_sanity.b); % DEBUG
        assert(abs(res_mean_irr_sanity.b(1)) < 1e-5);
        
        ix_reg{2} = ...
            W.Leverage.add_regressors( ...
                abs(Xs{dim_irr}(:, ix_t_incl) - bias), 'irr');
%             bsxfun(@times, ...
%                 Xs{dim_irr}(:, ix_t_incl) - bias, ...
%                 sign(ch_irr - 0.5)), ...
%                 'irr');
%             Xs{dim_irr}(tr_incl, 1:n_t_incl)); ...
        
        %% cond as RONI
        cond = W.Data.get_cond;
        cond = cond(tr_incl, :);
        
        cond_dim_incl = [];
        if length(unique(cond(:, dim_rel))) > 1
            cond_dim_incl = [cond_dim_incl, dim_rel];
        end
        if length(unique(cond(:, dim_irr))) > 1
            cond_dim_incl = [cond_dim_incl, dim_irr];
            
            res_cond_irr = glmwrap(cond(:, dim_irr), ch_irr, 'binomial');
            bias_cond_irr = -res_cond_irr.b(1) / res_cond_irr.b(2);
            cond(:, dim_irr) = abs(cond(:, dim_irr) - bias_cond_irr); ...
%                              .* sign(ch_irr - 0.5);
        end
        if ~isempty(cond_dim_incl)
            W.Leverage.add_regressors_of_no_interest(cond(:, cond_dim_incl));
        end

        %% Add interactions
        W.Leverage.add_ixn(ix_reg{:}, ch_irr);
        
        %% Regress
        W.Leverage.get_leverage;
        
        %% Specify reconstruction
        Bases_rel = W.Bases.Bases{dim_rel};
        W.Leverage.set_fun_recon(@Bases_rel.get_recon_y_and_t);
%         W.Leverage.LocalTime.set_dt(W.dt);
        
        %% Plot
        fig_tag('b');
        res.fig_tag = union(res.fig_tag, {'b'}, 'stable');
        clf;
        
        W.Leverage.plot_leverage;
        grid on;
        
        %% Plot ixn
        fig_tag('ixn');
        res.fig_tag = union(res.fig_tag, {'ixn'}, 'stable');
        clf;
        
        plot_ixn_opt = varargin2C({
            'shift_to_zero', true
            });
        
        W.Leverage.plot_ixn(plot_ixn_opt{:});
        grid on;
        axis auto;
        
%         fig_tag('ixn_irr_on_x');
%         cla;
%         W.Leverage.plot_ixn_irr_on_x(plot_ixn_opt{:});
%         grid on;
%         axis tight;
        
        %%
        fig_tag('prop_valid'); 
        res.fig_tag = union(res.fig_tag, {'prop_valid'}, 'stable');
        clf;
        
        W.Leverage.plot_prop_valid;
        grid on; 
        
        %% Regress
%         cla;
%         for kind = {'mean', 'max_possible'}
%             W.Leverage.get_leverage(kind{1});
% 
%             %% Plot
%             W.Leverage.plot_leverage;
%             hold on;
%         end

        %% Output
        
    end
end
%% Sanity check
methods
    function plot_wt_vs_cond(W)
        Bs = W.Bases.Bases;
        n_dim = W.Data.get_n_dim;
        cond = W.Data.get_cond;
        
        fig_tag('wt_vs_cond');
        for dim = 1:n_dim
            subplotRC(1, n_dim, 1, dim);
            
            c_cond = cond(:, dim);
            mean_wt = nanmean(Bs{dim}.wt, 2);
            cla; plot(c_cond, mean_wt, '.');
        end
    end     
end
%% Adaptors
methods
    function set_CosBases(W, CosBases)
        
    end
end
%% Set/Get
methods
    function set_Bases(W, obj_or_name)
        if nargin < 2, obj_or_name = Fit.D2.RT.En.EnCosBases; end
        obj_or_name = ...
            W.enforce_class('Fit.D2.RT.En.EnCosBases', ...
                obj_or_name);
        W.Bases = obj_or_name;
        W.set_sub_from_props({'Bases'});
    end
    function set_Leverage(W, obj_or_name)
        if nargin < 2, obj_or_name = zEn.EnLeverageGlmSep; end
        obj_or_name = ...
            W.enforce_class('zEn.EnLeverage', ...
                obj_or_name);
        W.Leverage = obj_or_name;
        W.set_sub_from_props({'Leverage'});
    end
    function set_Data(W, obj_or_name)
        if nargin < 2, obj_or_name = Fit.D2.Common.DataChRtPdfEn; end
        obj_or_name = ...
            W.enforce_class('Fit.D2.Common.DataChRtPdfEn', ...
                obj_or_name);
        W.set_Data@Fit.D2.Common.CommonWorkspace(obj_or_name);
    end
end
end