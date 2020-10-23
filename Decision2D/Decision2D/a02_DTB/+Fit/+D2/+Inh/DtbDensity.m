classdef DtbDensity < Fit.D2.Inh.Dtb
    % Fit.D2.Inh.DtbDensity
    %
    % To use KBRatio, simply call W.set_KBRatios;
    % To use KBRatio on only one of the dims, need to modify set_KBRatios.
    
    % 2015 YK wrote the initial version.
    
%% Settings
properties
    TO_DEBUG = 1; % 0: none, 1: warning, 2: error
    DEBUG_THRES = 1e-4; % 1e-10;
    VERBOSE_DEBUG = false;
    
    % Time to apply sign rule on 1st diffusion
    t_sign_rule_together_sec = {[], []};
    
    % Time to apply sign rule on 2nd diffusion
    t_sign_rule_alone_sec = {[], []};
    
    % Whether to slow bound collapse for deprioritized dimension
    % slow_collapse{dim} = kind
    % 'ssq': speed of collapse follows sigmaSq_fac_together.
    % 'none': Do not slow bound collapse.
    %         The bound collapses even if the drift/diffusion is not
    %         going on.
    to_slow_collapse = {'', ''};
%     to_slow_collapse = {'ssq', 'ssq'};
end
properties (Dependent)
    fix_fano
end
%% Intermediate variables
properties (Transient)
    % {dim}(t, cond, ch) = p(t, ch | cond, dim)
    td_together 
    
    % {dim}(t, y, cond) = p(t, y | cond, dim)
    unabs_together 
    
    % {dim}(t, cond1, cond2, c_ch) = p(t, c_ch, dim | cond1, cond2)
    td_together_first 
    
    % {dim}(t, y, cond1, cond2, o_ch) = p(t, y, o_ch | cond1, cond2, dim)
    unabs_together_first 
    
    % {dim_1st}(
    td_diff_ch
    
    % {dim_2nd}(t, cond1, cond2, c_ch, o_ch) = p(t, c_ch, o_ch | cond1, cond2)
    td_alone
    
    % {dim}(t, y, cond1, cond2, o_ch) = p(t, y, o_ch | cond1, cond2)
    unabs_alone
    
    % (t, cond1, cond2, ch1, ch2) = p(t, ch1, ch2 | cond1, cond2)
    td_merged
end
properties (Constant)
    dim_td_together = names2enum({'t', 'cond', 'ch'});
    dim_unabs_together = names2enum({'t', 'y', 'cond'});
    dim_td_together_first = names2enum({'t', 'cond1', 'cond2', 'ch'});
    dim_unabs_together_first = names2enum({'t', 'y', 'cond1', 'cond2', 'o_ch'});
    dim_unabs_alone = names2enum({'t', 'y', 'cond1', 'cond2', 'o_ch'});
    dim_td_merged = names2enum({'t', 'cond1', 'cond2', 'ch1', 'ch2'});
end
%% Construction
methods
    function W = DtbDensity(varargin)
        W.fix_sigmaSq_st = true;
        W.add_deep_copy({'Drift1', 'Drift2', 'Bound1', 'Bound2'});
        
        W.set_Drifts;
        W.set_Bounds;
        
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function set.fix_fano(W, to_fix_fano)
        is_fix_fano_already = isfield(W.th, 'drift_sigmaSq_fac_dim1_1');
        if to_fix_fano && ~is_fix_fano_already
            switch to_fix_fano
                case 1
                    fac_max = 1;
                    fac_min = 1;
                case 2
                    fac_max = 1;
                    fac_min = 0.16;
                otherwise
                    error('fix_fano=%d is not supported!\n', to_fix_fano);
            end
            W.add_params({
                {'drift_sigmaSq_fac_dim1_1', fac_max, 0.16, 1}
                {'drift_sigmaSq_fac_dim1_2', fac_min, 0.16, 1}
                {'drift_sigmaSq_fac_dim2_1', fac_min, 0.16, 1}
                {'drift_sigmaSq_fac_dim2_2', fac_max, 0.16, 1}
                });
            W.add_constraints({
                {'A', {'drift_sigmaSq_fac_dim1_2', ...
                       'drift_sigmaSq_fac_dim1_1'}, {[1, -1], 0}}
                {'A', {'drift_sigmaSq_fac_dim2_1', ...
                       'drift_sigmaSq_fac_dim2_2'}, {[1, -1], 0}}
                });
            for kind = {'drift', 'sigmaSq'}
                for d1 = 1:2
                    for d2 = 1:2
                        name = sprintf('%s_fac_together_dim%d_%d', ...
                            kind{1}, d1, d2);
                        W.fix_to_th0_(name);
                    end
                end
            end
        elseif ~to_fix_fano && is_fix_fano_already
            W.remove_params({
                'drift_sigmaSq_fac_dim1_1'
                'drift_sigmaSq_fac_dim1_2'
                'drift_sigmaSq_fac_dim2_1'
                'drift_sigmaSq_fac_dim2_2'
                });
            
        end
    end
    function v = get.fix_fano(W)
        v = isfield(W.th, 'drift_sigmaSq_fac_dim1_1');
    end
end
%% Prediction
methods
    function pred(W)
        W.init_Td_intermediate; % No effect
                                % - esp., doesn't fix parallel prediction.

        W.calc_fix_fano;
        W.calc_KBRatio;

        %%
        n_dim = 2;
        td_alone = cell(1, n_dim);
        
        %%
        for dim_1st = W.get_dim_1st_incl
            %%
            [td_together, unabs_together] = W.calc_Td_together(dim_1st);
            W.td_together = td_together;
            W.unabs_together = unabs_together;
            
            %%
            [td_alone{dim_1st}, unabs_alone] = W.calc_Td_final( ...
                td_together, unabs_together);
            
            W.td_alone = td_alone;
            W.unabs_alone = unabs_alone;
        end
        
        %%
        td_merged = W.calc_Td_merge_dim_1st(td_alone);
        
        %%
        n_ch = 2;
        W.Data.set_Td_pred_pdf( ...
            reshape(permute(td_merged, [1, 4, 5, 2, 3]), ...
                [W.nt, W.Data.get_nConds, n_ch, n_ch]));
    end
    function init_Td_intermediate(W)
        % TODO: init with appropriate size
        W.td_together = cell(1, 2);
        W.unabs_together = cell(1, 2);
        W.td_together_first = cell(1, 2);
        W.unabs_together_first = cell(1, 2);
        W.unabs_alone = cell(1, 2);
        W.td_alone = cell(1, 2);
        W.td_merged = [];
    end
    function calc_fix_fano(W)
        if W.fix_fano
            for d1 = 1:2
                for d2 = 1:2
                    drift_sigmaSq = sprintf('drift_sigmaSq_fac_dim%d_%d', ...
                        d1, d2);
                    v = W.th.(drift_sigmaSq);
                    
                    for kind = {'drift', 'sigmaSq'}
                        name = sprintf('%s_fac_together_dim%d_%d', ...
                            kind{1}, d1, d2);
                        W.th.(name) = v;
                        W.fix_to_th_(name);
                    end
                end
            end
        end
    end
end
%% SigmaSq_st
properties (Dependent)
    fix_sigmaSq_st % Whether to model starting point variability
    sigmaSq_st % (1, dim)
end
properties (Constant)
    MIN_SIGMASQ_ST = 1e-6;
end
methods
    function set.fix_sigmaSq_st(W, v)
        if v
            v0 = log10(W.MIN_SIGMASQ_ST);
            W.add_params({
                {'log10_sigmaSq_st_1', v0, v0, v0}
                {'log10_sigmaSq_st_2', v0, v0, v0}
                });
        else
            W.add_params({
                {'log10_sigmaSq_st_1', -1, -2, 0}
                {'log10_sigmaSq_st_2', -1, -2, 0}
                });
        end
    end
    function v = get.fix_sigmaSq_st(W)
        if isfield(W.th_fix, 'log10_sigmaSq_st_1')
            v = W.th_fix.log10_sigmaSq_st_1;
        else
            v = [];
        end
    end
    function v = get.sigmaSq_st(W)
        v = 10 .^ ...
            [W.th.log10_sigmaSq_st_1, W.th.log10_sigmaSq_st_2];
    end
    function set.sigmaSq_st(W, v)
        if isscalar(v)
            v = [v, v];
        end
        W.th.log10_sigmaSq_st_1 = log10(v(1));
        W.th.log10_sigmaSq_st_2 = log10(v(2));
    end
    function v = get_file_fields0(W)
        v = union_general( ...
            W.get_file_fields0@Fit.D2.Inh.Dtb, {
                'fix_sigmaSq_st', 'fsqs'
            }, 'stable', 'rows');
    end
end
%% Calculation
%% ----- calc_Td_together and its submethods
methods
    function [td_together, unabs_together] = calc_Td_together(W, dim_1st)
        %%
        n_dim = Data.Consts.n_dim;
        
        for dim = n_dim:-1:1
            [td_together{dim}, unabs_together{dim}] = ...
                W.calc_Td_together_unit(dim, dim_1st);
        end
    end
    function [td_pdf, unabs] = calc_Td_together_unit(W, dim, dim_1st)
        % td_pdf and unabs of dim when dim_1st is prioritized
        %
        % [td_pdf, unabs] = calc_Td_together_p(W, dim, dim_1st)
        %
        % td_pdf(t, cond, ch) = P(t, ch | cond)
        % unabs(t, y, cond) = P(t, y | cond)
        
        %% Get local variables
        t = W.get_t;
        y = W.get_y;
        
        drift_cond_t = W.get_drift_cond_t_together(dim, dim_1st);
        sigmaSq = W.get_sigmaSq_together(dim, dim_1st);
        sigmaSq_st = W.sigmaSq_st(dim);
        
        Bound = W.get_Bound(dim);   
        bound_t_ch = Bound.get_bound_t_ch;

        %% Process
        % Sign rule
        t_sign_rule = W.get_t_sign_rule_sec('together', dim);

        %% calc dtb
        % [(t, cond, ch), (t, y, cond)]
        [td_pdf, unabs] = ...
            W.calc_dtb( ...
                drift_cond_t, t, bound_t_ch, y, ...
                'sigmaSq', sigmaSq, ...
                'sigmaSq_st', sigmaSq_st, ...
                'is_constant_bound', false, ...
                'apply_sign_rule', false, ... true, ...
                't_sign_rule_sec', t_sign_rule);
    end
    function drift_cond_t = get_drift_cond_t_together(W, dim, dim_1st)
        % drift_cond_t(cond, t)
        
        drift_fac = W.get_drift_fac_together(dim, dim_1st);
        Drift = W.get_Drift(dim);
        drift_cond_t = Drift.get_drift_cond_t;
        drift_cond_t = bsxfun(@times, drift_cond_t, drift_fac);
    end
    function sigmaSq = get_sigmaSq_together(W, dim, dim_1st)
        % sigmaSq = get_sigmaSq_together(W, dim, dim_1st)
        %
        % sigmaSq: scalar or (cond, 1)
        
        sigmaSq = W.get_sigmaSq_fac_together(dim, dim_1st);
    end
end
%% ----- calc_Td_together_final
methods
    function [td_final, unabs_alone] = calc_Td_final(W, td_together, unabs_together)
        % td_final{dim}(t, cond12, ch_dim)
        % unabs_alone{dim}(t, y, cond12)
        
        %% -- Input
        n_dim = 2; % CONST
        
        %% td_same_ch
        % (t, ch1, ch2, cond12)
        % = P(R1 = t, R2 = t, Ch1 = ch1, Ch2 = ch2)
        
        td_same_ch = bsxfun(@times, ...
            permute(td_together{1}, [1, 3, 4, 2]), ...
            permute(td_together{2}, [1, 4, 3, 2]));
        
        %% unabs_diff_ch
        % {dim}(t, y_o_dim, ch_dim, cond12)
        % = P(R_dim = t, Ch_dim = ch, R_o_dim > t, Y_o_dim = y)
        
        for dim = n_dim:-1:1
            o_dim = n_dim + 1 - dim;
            unabs_diff_ch{dim} = bsxfun(@times, ...
                permute(td_together{dim},      [1, 4, 3, 2]), ...
                permute(unabs_together{o_dim}, [1, 2, 4, 3]));
%             siz = size(unabs_diff_ch{dim});
%             unabs_diff_ch{dim} = cat(1, ...
%                 zeros([1, siz(2:end)]), ...
%                 unabs_diff_ch{dim}(1:(end-1),:,:,:));
        end
        
        %% td_diff_ch
        % {dim}(t, ch_dim, ch_o_dim, cond12)
        % = P(R_dim<t, R_o_dim=t, Ch_dim=ch_dim, Ch_o_dim=ch_o_dim) 
        for dim = n_dim:-1:1
            o_dim = n_dim + 1 - dim;
            
            p0 = unabs_diff_ch{dim};
            [td_diff_ch{dim}, unabs_alone{dim}] = ...
                W.calc_Td_alone_unit(o_dim, p0);
        end
        W.td_diff_ch = td_diff_ch;
        
        %% td_final
        % (t, ch1, ch2, cond12)
        % = P(max(R1, R2)=t, ch1, ch2)
        td_final = td_same_ch ...
                 + td_diff_ch{1} ...
                 + permute(td_diff_ch{2}, [1, 3, 2, 4]);
    end
end
%% ----- calc_Td_alone_unit
methods
    function [td_alone, unabs_alone] = calc_Td_alone_unit(W, o_dim, p0)
        % p0_dim(t, y_o_dim, ch_dim, cond12)
        % = P(R_dim = t, Ch_dim = ch, R_o_dim > t, Y_o_dim = y)
        %
        % td_alone(t, ch_dim, ch_o_dim, cond12)
        % unabs_alone(t, o_y, ch_dim, cond12)
        
        %% Input
        Bound = W.get_Bound(o_dim);
        bound_t_ch = Bound.get_bound_t_ch;

        drift_cond_t = W.get_drift_cond_t_dim(o_dim);            
        sigmaSq = W.get_sigmaSq_alone(o_dim);

        t = W.get_t;
        y = W.get_y;

        %% Loop
        n_ch = 2;
        for ch = n_ch:-1:1
            % unabs(y_o_dim, cond12, t) % given ch_dim
            unabs1 = permute(p0(:, :, ch, :), [2, 4, 1, 3]);

            % Prepare for applying sign rule
            t_sign_rule = W.get_t_sign_rule_sec('alone', o_dim);

            %% -- Process
            % [(t, cond12, o_ch), (t, o_y, cond12)]
            [td1, unabs_alone1] = W.calc_dtb( ...
                    drift_cond_t, t, bound_t_ch, y, ...
                    'p0', unabs1, ...
                    'sigmaSq', sigmaSq, ...
                    'apply_sign_rule', false, ... true, ...
                    't_sign_rule_sec', t_sign_rule);

            %% -- Output
            % new format:
            % td_alone(t, ch_dim, ch_o_dim, cond12)
            td_alone(:, ch, :, :) = ...
                permute(td1, [1, 4, 3, 2]);
            
            unabs_alone(:, :, ch, :) = ...
                permute(unabs_alone1, [1, 2, 4, 3]);
        end
    end
    function drift_cond_t = get_drift_cond_t_dim(W, dim)
        n_conds = W.Data.get_nConds;
        n_dim = 2; % CONST
        
        [ix_conds{1:n_dim}] = ndgrid(1:n_conds(1), 1:n_conds(2));
        for dim = 1:n_dim
            ix_conds{dim} = ix_conds{dim}(:);
        end
        
        Drift = W.get_Drift(dim);
        
        drift_cond_t = Drift.get_drift_cond_t;
        drift_cond_t = drift_cond_t(ix_conds{dim}(:), :);
    end
    function sigmaSq = get_sigmaSq_alone(W, dim)
        sigmaSq = 1;
    end
end
%% ----- calc_Td_merged and its submethods
methods
    function td_merged = calc_Td_merge_dim_1st(W, td_final)
        dim_1st_incl = W.get_dim_1st_incl;
        if length(dim_1st_incl) == 2        
            p_dim1_1st = W.th.p_dim1_1st;
            td_merged = td_final{1} * p_dim1_1st ...
                      + td_final{2} * (1 - p_dim1_1st);
            
        else
            assert(isscalar(dim_1st_incl));
            td_merged = td_final{dim_1st_incl};
        end
    end
end
%% Prediction - Utility
methods
    function [td_pdf, unabs] = calc_dtb(~, drift_cond_t, t, bound_t_ch, y, ...
            varargin)
        % td_pdf(t, cond, ch) = p(t, ch | cond)
        % unabs(t, y, cond)  = p(t, y | cond)
        [td_pdf, unabs] = ...
            dtb.pred.calc_dtb( ...
                drift_cond_t, t, bound_t_ch, y, ...
                varargin{:});
            % TODO: Spectral just to match y with spectral.
            %       May still use analytic if y's are matched.
    end
    function t = get_t_sign_rule_sec(W, kind, dim)
        % kind : 'together' | 'alone'
        % dim : 1 or 2.
        switch kind
            case 'together'
                if isempty(W.t_sign_rule_together_sec{dim})
                    t = W.get_max_t;
                else
                    t = W.t_sign_rule_together_sec{dim};
                end
            case 'alone'
                if isempty(W.t_sign_rule_alone_sec{dim})
                    t = W.get_max_t;
                else
                    t = W.t_sign_rule_alone_sec{dim};
                end
            otherwise
                error('Unknown kind!');
        end
    end
    function ix = get_t_sign_rule_ix(W, kind, dim)
        t = W.get_t_sign_rule_sec(kind, dim);
        [~, ix] = min(abs(W.get_t - t));
    end
end
%% Plot mean RT & ch
methods
    function plot_ch_td(W, td_pdf, varargin)
        S = varargin2S(varargin, {
            'h', []
            });
        if nargin < 2
            td_pdf = W.Data.get_Td_pred_pdf;
        end
        if ndims(td_pdf) < 5
            td_pdf = repmat(td_pdf, [1 1 1 1 2]);
        end            
        
        n_dim = 2;
        nr = 2;
        nc = n_dim;
        if isempty(S.h)
            S.h = subplotRCs(nr, nc);
        else
            assert(isequal(size(S.h), [nr, nc]));
        end
        
        for dim = 1:n_dim
            axes(S.h(1, dim));
            Pl = DtbPlot.PlotCh2D(td_pdf, {'dimOnX', dim});
%             Pl.foldDim = [false, false];
            Pl.plot;
        end
        for dim = 1:n_dim
            axes(S.h(2,dim));
            Pl = DtbPlot.PlotRt2D(td_pdf, {'dimOnX', dim});
%             Pl.foldDim = [false, false];
            Pl.plot;
        end
    end
    function plot_ch_td_together_first(W)
        n_dim = 2;
        for dim = n_dim:-1:1
            td{dim} = repmat(W.td_together_first{dim}, [1, 1, 1, 1, 2]);
        end
        td{2} = permute(td{dim}, [1, 2, 3, 5, 4]);
        
        for dim = n_dim:-1:1
            fig_tag(str_con('ch_td_together', dim));
            W.plot_ch_td(td{dim});
        end
    end
end
%% Plot Td
methods
    function plot_td_ch2d(W, p, varargin)
        % p(t, ch1, ch2, cond12)
        
        
    end
    function plot_td(W, p, varargin)
        % p(t, ch, cond12)
        
        
        
        %%
        n_ch = 2;
        p = reshape(permute(p, [1, 3, 2]), ...
            [W.nt, W.Data.nConds, n_ch, n_ch]);
        
        Pl = DtbPlot.PlotRt2D(p);
    end
    function plot_td_together(W)
        td_pdfs = W.td_together;
        
        n_dim = 2;
        n_conds = W.Data.get_nConds;
        conds_all = W.Data.get_conds;
        
        fig_tag('td_together');
        clf;
        n_row = max(n_conds);
        n_col = n_dim;
        h = subplotRCs(n_row, n_col);
        
        for dim = n_dim:-1:1
            td_pdf = td_pdfs{dim};

            conds = conds_all{dim};
            titles = csprintf('dim=%d, cond=%1.3g', dim, conds);
            
            W.plot_td_pdf_cond_ch(td_pdf, ...
                'h', h(:,dim), ...
                'titles', titles);
        end
        legend(h(end), {'ch1', 'ch2'});
    end
    function plot_td_together_first(W)
        td_pdfs = W.td_together_first;
        
        nt = W.get_nt;
        n_ch = 2;
        n_dim = 2;
        n_conds = W.Data.get_nConds;
        conds_all = W.Data.get_conds;
        
        for dim = n_dim:-1:1
            td_pdf = td_pdfs{dim};
            td_pdf = reshape(td_pdf, nt, prod(n_conds), n_ch);
            
            fig_tag(str_con('td_together_first_dim', dim));
            clf;
            h = subplotRCs(n_conds(1), n_conds(2));
%             titles = csprintf('dim=%d, cond1=%1.3g, cond2=%1.3g', ...
%                 dim, conds_comb{1}, conds_comb{2});
            
            W.plot_td_pdf_cond_ch(td_pdf, ...
                'h', h);
            
            gltitle(h, 'row', csprintf('cond1=%1.3g', conds_all{1}));
            gltitle(h, 'col', csprintf('cond2=%1.3g', conds_all{2}));
            legend(h(end), {'ch1', 'ch2'});
        end
    end
    function plot_td_alone(W)
        td_pdfs = W.td_alone;
        
        nt = W.get_nt;
        n_ch = 2;
        n_dim = 2;
        n_conds = W.Data.get_nConds;
        conds_all = W.Data.get_conds;
        
        for dim = n_dim:-1:1
            for o_ch = n_ch:-1:1
                td_pdf = td_pdfs{dim}(:,:,:,:,o_ch);
                td_pdf = reshape(td_pdf, nt, prod(n_conds), n_ch);

                fig_tag(str_con('td_together_dim2nd', dim, 'o_ch', o_ch));
                clf;
                h = subplotRCs(n_conds(1), n_conds(2));
    %             titles = csprintf('dim=%d, cond1=%1.3g, cond2=%1.3g', ...
    %                 dim, conds_comb{1}, conds_comb{2});

                W.plot_td_pdf_cond_ch(td_pdf, ...
                    'h', h);

                gltitle(h, 'row', csprintf('cond1=%1.3g', conds_all{1}));
                gltitle(h, 'col', csprintf('cond2=%1.3g', conds_all{2}));
                legend(h(end), {'ch1', 'ch2'});
            end
        end
    end
    %% Plot Td - Internal
    function plot_td_pdf_cond_ch(W, td_pdf, varargin)
        % plot_td_pdf_cond_ch(W, td_pdf, ...)
        % 
        % td_pdf(t, cond, ch)
        
        S = varargin2S(varargin, {
            'h', []
            'titles', {}
            });
        
        % Input
        n_ch = 2;
        nt = W.get_nt;
        n_cond = size(td_pdf, 2);
        assert(isequal(size(td_pdf), [nt, n_cond, n_ch]));
        
        if ~isempty(S.titles)
            assert(iscell(S.titles));
            assert(all(cellfun(@ischar, S.titles)));
            assert(numel(S.titles) == n_cond);
        end

        if isempty(S.h)
            S.h = subplotRCs(n_cond, 1);
        else
            assert(all(isvalidhandle(S.h(:))));
            assert(numel(S.h) >= n_cond);
        end

        t = W.get_t;
        
        % Process
        for i_cond = n_cond:-1:1
            plot(S.h(i_cond), t, squeeze(td_pdf(:, i_cond, :)));
            yLim = ylim;
            ylim([0, yLim(2)]);
            
            if ~isempty(S.titles)
                title(S.h(i_cond), S.titles{i_cond});
            end
        end
        sameAxes(S.h);        
    end
end
%% Plot unabsorbed
methods
    function plot_unabs_together(W)
        unabses = W.unabs_together;
        
        n_dim = 2;
        conds_all = W.Data.get_conds;
        n_conds = W.Data.get_nConds;
        
        fig_tag('unabs_together');
        clf;
        n_row = max(n_conds);
        n_col = n_dim;
        h = subplotRCs(n_row, n_col);
        
        for dim = n_dim:-1:1
            unabs = unabses{dim};
            
            conds = conds_all{dim};
            titles = csprintf('dim=%d, cond=%1.3g', dim, conds);
            
            W.plot_unabs_cond(unabs, ...
                'h', h(:, dim), ...
                'titles', titles);
        end
    end
    function plot_unabs_together_first(W)
        unabses = W.unabs_together_first;
        
        nt = W.get_nt;
        ny = W.get_ny;
        n_ch = 2;
        n_dim = 2;
        n_conds = W.Data.get_nConds;
        conds_all = W.Data.get_conds;
        
        for dim = n_dim:-1:1
            unabs_chs = unabses{dim};
            unabs_chs = reshape(unabs_chs, nt, ny, prod(n_conds), n_ch);
            for o_ch = n_ch:-1:1
                unabs = unabs_chs(:,:,:,o_ch);

                fig_tag(str_con('unabs_together_first_dim', dim, 'o_ch', o_ch));
                clf;
                h = subplotRCs(n_conds(1), n_conds(2));

                W.plot_unabs_cond(unabs, ...
                    'h', h);
            
                gltitle(h, 'row', csprintf('cond1=%1.3g', conds_all{1}));
                gltitle(h, 'col', csprintf('cond2=%1.3g', conds_all{2}));
            end
        end
    end
    function plot_unabs_alone(W)
        % Copied from plot_unabs_together_first
        unabses = W.unabs_alone;
        
        nt = W.get_nt;
        ny = W.get_ny;
        n_ch = 2;
        n_dim = 2;
        n_conds = W.Data.get_nConds;
        conds_all = W.Data.get_conds;
        
        for dim = n_dim:-1:1
            unabs_chs = unabses{dim};
            unabs_chs = reshape(unabs_chs, nt, ny, prod(n_conds), n_ch);
            for o_ch = n_ch:-1:1
                unabs = unabs_chs(:,:,:,o_ch);

                fig_tag(str_con('unabs_alone', dim, 'o_ch', o_ch));
                clf;
                h = subplotRCs(n_conds(1), n_conds(2));

                W.plot_unabs_cond(unabs, ...
                    'h', h);
            
                gltitle(h, 'row', csprintf('cond1=%1.3g', conds_all{1}));
                gltitle(h, 'col', csprintf('cond2=%1.3g', conds_all{2}));
            end
        end        
    end
    function plot_unabs_cond(W, unabs, varargin)
        % plot_unabs_cond(W, unabs, ...)
        % 
        % unabs(t, y, cond)
        
        S = varargin2S(varargin, {
            'h', []
            'titles', {}
            });
        
        % Input
        nt = W.get_nt;
        ny = W.get_ny;
        n_cond = size(unabs, W.dim_unabs_together.cond);
        assert(isequal(size(unabs), [nt, ny, n_cond]));
        
        if ~isempty(S.titles)
            assert(iscell(S.titles));
            assert(all(cellfun(@ischar, S.titles)));
            assert(numel(S.titles) == n_cond);
        end

        if isempty(S.h)
            S.h = subplotRCs(n_cond, 1);
        else
            assert(all(isvalidhandle(S.h(:))));
            assert(numel(S.h) >= n_cond);
        end
        
        t = W.get_t;
        y = W.get_y;
        
        % Process
        for i_cond = n_cond:-1:1
            c_unabs = max(unabs(2:end,2:(end-1),i_cond)', eps); %#ok<UDIM>
            axes(S.h(i_cond)); %#ok<LAXES>
            imagesc(t, y, c_unabs);
            
            if ~isempty(S.titles)
                title(S.h(i_cond), S.titles{i_cond});
            end
        end
    end
end
%% Demo
methods (Static)
    function W = demo
        %%
        init_path;
        W = Fit.D2.Inh.DtbDensity;
        W.Data.set_path;
        W.Data.load_data;
        W.pred;
        W.plot_ch_RT;
    end
    function W = demo_par
        W = Fit.D2.Inh.DtbDensity;
        W.Data.set_path;
        W.Data.load_data;
        %%
        W.th.Drift1__k = 20;
        W.th_numbered.drift_fac_together_dim = [1 1; 1 1];
        W.th_numbered.sigmaSq_fac_together_dim = [1 1; 1 1];
        W.pred;
        W.plot_ch_RT;
    end
    function W = demo_ser
        W = Fit.D2.Inh.DtbDensity;
        W.Data.set_path;
        W.Data.load_data;
        %%
        W.th.Drift1__k = 20;
        W.th_numbered.drift_fac_together_dim = [1 0; 0 1];
        W.th_numbered.sigmaSq_fac_together_dim = [1 0.04; 0.04 1];
        W.pred;
        W.plot_ch_RT;
    end
    function W = demo_max_t(W)
        if nargin < 1
            W = Fit.D2.Inh.DtbDensity.demo_par;
        end
        
        %%
        fig_tag('demo_max_t');
        
        %%
%         nr = 2;
%         nc = 1;
        dim = 1;
        cond1 = 3;
        cond2 = 3;
        ch1 = 1;
        ch2 = 1;
        
        %% Td
        subplot('position', [0.1 0.5 0.8 0.4]);
%         subplotRC(nr, nc, 1, 1);
        cla;
        f = @(v) (v) / sum(v);
        t = W.get_t;
        y = W.get_y;
        
        p_together = [W.td_together{1}(:,cond1,ch1), ...
                      W.td_together{2}(:,cond2,ch2)];
        p_max = max_distrib(p_together);
        
        plot(t, f(p_together(:,1)), 'b:');
        hold on;
        
        plot(t, f(p_together(:,2)), 'b--');
        
        plot(t, f(p_max), 'b-')
        
        p_together_first = W.td_together_first{dim}(:, cond1, cond2, ch2);
        plot(t, f(p_together_first), 'm-');
        
        td_alone = W.td_alone{dim}(:, cond1, cond2, ch2);
        plot(t, f(td_alone), 'r--');
        
        td_pred = W.Data.get_Td_pred_pdf;
        plot(t, f(td_pred(:,cond1,cond2,ch1,ch2)), 'r-');
        
        % Expected: b- and r-- and r- and bo (and perhaps g-) should look similar.
        % Especially, b- and r- should match.
        
        axis tight;
        yLim = ylim;
        ylim([0, yLim(2)]);
        
        % Unabs
        p_unabs = W.unabs_alone{dim}(:,:,cond1,cond2,ch2);
        dim_y = 2;
%         subplotRC(nr, nc, 1, 1);
        p0_unabs = sum(p_unabs, dim_y);
        plot(t, f(p0_unabs), 'g-');
        
        iy = find(y > -1, 1, 'first');
        
        plot(t, f(p_unabs(:,iy)), 'bo');
        
        %% Unabs
        subplot('position', [0.1 0.1 0.8 0.4]);
%         subplotRC(nr, nc, 2, 1);
        cla;
        imagesc(t, y, p_unabs');
        ylim([-1, 1]);
        
    end
end
end