classdef Ev2PchD1 ...
        < IxnKernel.EvTime.CommonWorkspace
    % IxnKernel.RevCor.D1.Ev2PchD1 - Converts evidence to Pch
    
%% Settings
properties
%     Ev = IxnKernel.EvTime.EvTimeD1; % Inherited from IxnKernel.EvTime.CommonWorkspace
end
%% Results
properties
    % res_beta
    % .rt_incl_ms
    % .align
    % .itv_len_ms
    % .t_st_ms(1,t)
    % .slope(1,t)
    % .bias(1,t)
    res_beta = struct;
end
%% Init
methods
    function EP = Ev2PchD1(varargin)
        if nargin > 0
            EP.init(varargin{:});
        end
    end
    function init(EP, varargin)
        varargin2props(EP, varargin, true);
    end
    function main(EP)
        EP.batch_fit_intervals;
        EP.plot_and_save_all;
    end
end
%% Fit
methods
    function batch_fit_intervals(EP, varargin)
        S_batch = varargin2S(varargin, {
            'align', {'st'} % 'rt', 'st'}
            });
        [Ss, n] = factorizeC(S_batch);
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            
            EP.fit_intervals(C{:});
        end
    end
    function res_beta = fit_intervals(EP, varargin)
        C0 = varargin2C(varargin, {
            'interval_ix', 1:10
            'interval_spacing_ms', 100
            });
        
        %
        S0_filt = EP.Ev.get_S_filt(C0{:});
        itv_ix = S0_filt.interval_ix;
        n_itvs = numel(itv_ix);
        
        for ii = n_itvs:-1:1
            itv_ix1 = itv_ix(ii);
            
            C = varargin2C({
                'interval_ix', itv_ix1
                }, C0);
            
            [ev, ch, S_filt] = EP.Ev.get_ev_filtered(C{:});
            [slope(1,ii), intercept(1,ii), bias(1,ii), stats1] = ...
                EP.fit_interval(ev, ch);
            stats(1,ii) = rmfield(stats1, {
                'resid', 'residp', 'residd', 'resida', 'wts'
                });
        end
        
        % Set to res_beta
        res_beta = S_filt;
        res_beta.t_st_ms = S0_filt.interval_offset_ms ...
                       + S0_filt.interval_spacing_ms ...
                       * (S0_filt.interval_ix - 1);
        res_beta = copyFields(res_beta, ...
            packStruct(slope, bias, intercept, stats));
        
        if nargout == 0
            EP.res_beta = res_beta;
        end
    end
    function [slope, intercept, bias, stats] = fit_interval(~, ev, ch)
        % [slope, intercept, bias, stats] = fit_interval(~, ev, ch)
        
        mev = cellfun(@nanmean, ev);
        
        [b, ~, stats] = glmfit(mev, ch, 'binomial');
        slope = b(2);
        intercept = b(1);
        bias = -b(1) / b(2);
    end
    function f = get_f_p_ch_given_util(EP)
        assert(~isempty(EP.res_beta));
        f = @(ev, itv) EP.f_p_ch_given_util(ev, itv, ...
            EP.res_beta);
    end
    function p_ch_given_util = f_p_ch_given_util(EP, ev, n_used, res_beta)
        % p_ch_given_util = f_p_ch_given_util(EP, ev, n_used, res_beta)
        % 
        % p_ch_given_util(tr,itv) = Pr(ch(tr)==1 | ev(tr,itv) is used)
        %
        % p_ch_given_util(tr, itv)
        % ev(tr, 1)
        % n_passed_minus_1(1, itv)
        
        nu1 = n_used + 1;
        
        if isscalar(nu1)
            p_ch_given_util = glmval( ...
                [res_beta.intercept(nu1); res_beta.slope(nu1)], ...
                ev, 'logit');
        else
            assert(isrow(nu1));
            for ii = numel(nu1):-1:1
                p_ch_given_util(:,ii) = EP.f_p_ch_given_util(ev, nu1(ii), res_beta);
            end
        end
    end
end
%% Plot
methods
    function plot_and_save_all(EP)
        siz = [300, 250];
        
        %%
        clf;
        EP.imagesc_ev_w_rt;
        file = EP.get_file({
            'plt', 'ev_w_rt'
            });
        savefigs(file, 'size', siz);
        
        %%
        clf;
        [~, file] = EP.plot_ch_t_intervals('interval_step', 4);
        title('');
        savefigs(file, 'size', siz);
        
        %%
        EP2 = deep_copy(EP);
        EP2.fit_intervals('interval_ix', 1:2:10);
        clf;
        [~, file] = EP2.plot_slope_by_t;
        title('');
        savefigs(file, 'size', siz);
        
        %%
        EP.plot_ch_mixture_batch;
    end
    function imagesc_ev_w_rt(EP, varargin)
        S = varargin2S(varargin, {
            'rt_incl_ms', [0 1000]
            });
        
        %%
        ev = EP.Ev.ev;
        rt = (EP.Ev.rt_fr - 1) .* EP.dt * 1e3;
        tr_incl = (S.rt_incl_ms(1) <= rt) & (rt <= S.rt_incl_ms(2));
        ev = ev(tr_incl, :);
        rt = rt(tr_incl);
        
        [rt, ix] = sort(rt);
        ev = ev(ix,:);
        tr = 1:numel(rt);
        
        %%
        cla;
        imagesc(EP.t * 1e3, tr, ev);
        hold on;
        plot(rt, tr, 'ks', ...
            'MarkerSize', 1.5, ...
            'MarkerFaceColor', 'k', ...
            'MarkerEdgeColor', 'none');
        hold off;
        
        xlim(S.rt_incl_ms);
        
        ylabel('Trial');
        xlabel('Time (ms)');
        bml.plot.beautify;
    end
    function [h, file] = plot_slope_by_t(EP, varargin)
        S = varargin2S(varargin2S(varargin, EP.res_beta), {
            't_st_ms', {[]}
            'slope', {[]}
            'y', 'slope' % slope | bias | intercept
            'plot_args', {'k-'}
            'style', 'errorbarShade' % 'errorbarShade'|'line'
            });
        
        t = S.t_st_ms;
        y = S.(S.y);
        
        switch S.style
            case 'line'
                h.line = plot(t, y, S.plot_args{:});
            case 'errorbarShade'
                se = [S.stats.se];
                
                switch S.y
                    case 'slope'
                        e = se(2,:);
                    case 'intercept'
                        e = se(1,:);
                    otherwise
                        error('y=%s is not supported!', S.y);
                end
                [h.line, h.patch] = ...
                    errorbarShade(t, y, e, 'k-');
        end
        bml.plot.beautify;
        
        xlabel('Time (ms)');
        ylabel([bml.str.upper(S.y, 'sentence'), ' (a.u.)']);
        
        C = {'align', S.align, 'rt', S.rt_incl_ms, 'plt', 'slt'};
        title(EP.get_title(C));
        file = EP.get_file(C);
    end
    function varargout = plot_ch_t_st(EP, varargin)
        [varargout{1:nargout}] = EP.plot_ch_t_intervals('align', 'st');
    end
    function varargout = plot_ch_t_rt(EP)
        [varargout{1:nargout}] = EP.plot_ch_t_intervals('align', 'rt');
    end
    function plot_ch_mixture_batch(EP)
        itvs = 0:4:10;
        n_itv = numel(itvs);
        colors = hsv2(n_itv);
        
        for i_itv = 1:n_itv
            itv = itvs(i_itv);
            color = colors(i_itv, :);
            [~, file] = EP.plot_ch_mixture( ...
                'itv', itv, ...
                'color_used', color);
            title('');
            savefigs(file, 'size', [200, 150]);
        end
    end
    function [h_lines, file] = plot_ch_mixture(EP, varargin)
        S = varargin2S(varargin, {
            'n_mix', 3
            'itv', 0 % interval
            'color_used', [1 0 0]
            'color_unused', [0 0 0]
            });
        p_used = linspace(0, 1, S.n_mix);
        colors = linspaceN(S.color_unused, S.color_used, S.n_mix);
        
        itv = S.itv + 1;
        
        h_lines = gobjects(S.n_mix, 1);
        
        for i_mix = 1:S.n_mix
            p_used1 = p_used(i_mix);
            
            hs = EP.plot_ch_t_intervals( ...
                'to_add_legend', false, ...
                'to_plot_data', false, ...
                'interval_ix', itv, ...
                'p_used', p_used1);
            hold on;
            set([hs.pred], ...
                'Color', colors(i_mix, :), ...
                'LineWidth', 2);
            h_lines(i_mix) = hs.pred;
        end
        hold off;
        legend(flip(h_lines), ...
            csprintf('p_{acq}=%1.1f', flip(p_used)), ...
            'Location', 'NorthWest');
        
        file = EP.get_file({
            'n_mix', S.n_mix
            'itv', S.itv
            });
    end
    function [hs, file] = plot_ch_t_intervals(EP, varargin)
        S = varargin2S(varargin, {
            'interval_ix', []
            'interval_step', 1
            'to_add_legend', true
            });
        
        if isempty(S.interval_ix)
            S = rmfield(S, 'interval_ix');
        end
        S_filt = EP.Ev.get_S_filt_intervals(varargin{:});
        
        n_interval = numel(S_filt.interval_ix);
        interval_incl = 1:S.interval_step:n_interval;
        
        n_interval_plot = numel(interval_incl);
        colors = hsv2(n_interval_plot);
        legends = cell(n_interval_plot, 1);
        
        clear hs
        
        for i_itv = 1:n_interval_plot
            
            itv = interval_incl(i_itv);
            C = varargin2C({
                'interval_ix', S_filt.interval_ix(itv)
                'color', colors(i_itv, :)
                }, S_filt);
            
            hs(i_itv) = EP.plot_ch_t_interval(C{:}); %#ok<AGROW>
            hold on;
            
            [~, st_sec_rel, en_sec_rel] = EP.Ev.get_fr_incl(C{:});
            
            legends{i_itv} = sprintf('%1.0f-%1.0f ms', ...
                st_sec_rel * 1e3, en_sec_rel * 1e3);
        end
        hold off;
        
        if S.to_add_legend
            if isvalidhandle(hs(1).pred)
                set([hs.pred], 'LineWidth', 2);
                legend([hs.pred], legends, 'Location', 'SouthEast');
            else
                legend([hs.data], legends, 'Location', 'SouthEast');
            end
        end
        
        C = {
            'plt', 'ch'
            'align', S_filt.align
            'rt', S_filt.rt_incl_ms
            };
        title(EP.get_title(C));
        file = EP.get_file(C);
        
        h = struct; % Not used now
    end
    function [h, file] = plot_ch_t_interval(EP, varargin)
        [ev, ch] = EP.Ev.get_ev_filtered(varargin{:});
        [h, file] = EP.plot_ch('ev', ev, 'ch', ch, varargin{:});
%         disp(varargin2S(varargin));
    end
    function [h, file] = plot_ch(EP, varargin)
        %%
        S = varargin2S(varargin, {
            'n_bin', 11
            'ev', EP.Ev.ev
            'ch', EP.Ev.ch
            'color', 'k'
            'to_plot_data', true
            'to_plot_fit', true
            'p_used', 1
            });
        
        if iscell(S.ev)
            n_tr = size(S.ev, 1);
            mev = zeros(n_tr, 1);
            for tr = 1:n_tr
                mev(tr) = nanmean(S.ev{tr});
            end
%             mev = cellfun(@(v) nanmean(v, 2), S.ev);
%             mev = nanmean(cell2mat2(S.ev), 2);
        else
            assert(isnumeric(S.ev));
            mev = nanmean(S.ev, 2);
        end
        
        % NaN can occur if all(isnan(ev(tr,:))).
        tr_incl = ~isnan(mev); 
        [d_cond, ~, conds] = quantilize(mev, S.n_bin);
        
        % Binned p_ch for plotting
        d_cond = d_cond(tr_incl);
        ch = S.ch(tr_incl);
        p_ch = accumarray(d_cond, ch, [], @nanmean);
        
        % Fit if plotting
        h = struct;
        if S.to_plot_fit
            b = glmfit(mev, ch, 'binomial');
            conds_fit = linspace(conds(1), conds(end));
            y_fit = glmval(b, conds_fit, 'logit');
            
            y_fit = (y_fit - 0.5) .* S.p_used + 0.5;
            
            h.pred = plot(conds_fit, y_fit, '-', 'Color', S.color);
            hold on;
        else
            h.pred = gobjects;
        end
        
        % Plot data
        if S.to_plot_data
            h.data = plot(conds, p_ch, ...
                'o', ...
                'MarkerFaceColor', S.color, ...
                'MarkerEdgeColor', 'w');
        else
            h.data = gobjects;
        end
        hold off;
        ylim([0, 1]);
        
        % Beautify
        bml.plot.beautify;
        bml.plot.axis_margin;
        x_ticks = get(gca, 'XTick');
        set(gca, ...
            'XTick', [x_ticks(1), 0, x_ticks(end)], ...
            'YTick', [0, 0.5, 1]);
        xlabel('Mean evidence');
        ylabel('P_{ch}');
        
        % Outputs
        file = EP.get_file({'plt', 'ch'});
    end
    function [h, file] = plot_ecdf_rt(EP)
        [f, x] = ecdf(EP.Ev.rt_fr);
        stairs(EP.t(x), f, 'k-');
        set(gca, ...
            'XTick', 0:1:EP.max_t, ...
            'YTick', [0, 0.5, 1]);
        xlabel('t (s)');
        ylabel('F(Td < t)');
        bml.plot.beautify;
        
        h = struct;
        file = EP.get_file({'plt', 'ecdf_rt'});
    end
end
end