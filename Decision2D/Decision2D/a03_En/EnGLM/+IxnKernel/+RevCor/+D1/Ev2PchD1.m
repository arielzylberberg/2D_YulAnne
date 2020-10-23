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
%     function batch_fit_intervals(EP, varargin)
%         S_batch = varargin2S(varargin, {
%             'align', {'rt', 'st'}
%             });
%         [Ss, n] = factorizeC(S_batch);
%         for ii = 1:n
%             S = Ss(ii);
%             C = S2C(S);
%             
%             EP.fit_intervals(C{:});
%         end
%     end
    function res_beta = fit_intervals(EP, varargin)
        C0 = varargin2C(varargin, {
            'interval_ix', 1:10
            'interval_spacing_ms', 100
            });
        
        %
        S_filt = EP.Ev.get_S_filt(C0{:});
        itv_ix = S_filt.interval_ix;
        n_itvs = numel(itv_ix);
        
        for ii = n_itvs:-1:1
            itv_ix1 = itv_ix(ii);
            
            C = varargin2C({
                'interval_ix', itv_ix1
                }, C0);
            
            [slope(1,ii), intercept(1,ii), bias(1,ii)] = ...
                EP.fit_interval(C{:});
        end
        
        % Set to res_beta
        res_beta = S_filt;
        res_beta.t_st_ms = S_filt.interval_offset_ms ...
                       + S_filt.interval_spacing_ms ...
                       * (S_filt.interval_ix - 1);
        res_beta = copyFields(res_beta, ...
            packStruct(slope, bias, intercept));
        
        if nargout == 0
            EP.res_beta = res_beta;
        end
    end
    function [slope, intercept, bias, stats, S_filt] = fit_interval(EP, varargin)
        [ev, ch, S_filt] = EP.Ev.get_ev_filtered(varargin{:});
        
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
        for kind = {'ch', 'ecdf_rt', 'slope_by_t'}
            clf;
            [~, file] = EP.(['plot_' kind{1}]);
            savefigs(file);
        end
        
        % ch_t_intervals
        S_batch = varargin2S({
            'align', {'st', 'rt'}
            'rt_incl_ms', {[0, 5000], [1100, 5000]}
            });
        [Ss, n] = factorizeS(S_batch);
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            [~, file] = EP.plot_ch_t_intervals(C{:});
            savefigs(file);
        end
    end
    function [h, file] = plot_slope_by_t(EP, varargin)
        S = varargin2S(EP.res_beta, {
            't_st_ms', {[]}
            'slope', {[]}
            'y', 'slope'
            'plot_args', {}
            });
        
        t = S.t_st_ms;
        slope = S.(S.y);
        
        h = plot(t, slope, S.plot_args{:});
        bml.plot.beautify;
        
        xlabel('Time (ms)');
        ylabel(bml.str.upper(S.y, 'sentence'));
        
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
    function [h, file] = plot_ch_t_intervals(EP, varargin)
        S_filt = EP.Ev.get_S_filt_intervals(varargin{:});
        
        n_interval = numel(S_filt.interval_ix);
        colors = hsv2(n_interval);
        
        legends = cell(n_interval, 1);
        for i_itv = 1:n_interval
            C = varargin2C({
                'interval_ix', S_filt.interval_ix(i_itv)
                'plot_args', varargin2plot({
                    'Color', colors(i_itv, :)
                    'Marker', 'o'
                    'LineStyle', '-'
                    })
                }, S_filt);
            
            EP.plot_ch_t_interval(C{:});
            hold on;
            
            [~, st_sec_rel, en_sec_rel] = EP.Ev.get_fr_incl(C{:});
            
            legends{i_itv} = sprintf('%1.0f-%1.0f ms', ...
                st_sec_rel * 1e3, en_sec_rel * 1e3);
        end
        hold off;
        legend(legends, 'Location', 'SouthEast');
        
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
        ev = EP.Ev.get_ev_filtered(varargin{:});
        [h, file] = EP.plot_ch('ev', ev, varargin{:});
    end
    function [h, file] = plot_ch(EP, varargin)
        %%
        S = varargin2S(varargin, {
            'n_bin', 9
            'ev', EP.Ev.ev
            'ch', EP.Ev.ch
            'plot_args', {'ko-'}
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
        p_ch = accumarray(d_cond(tr_incl), S.ch(tr_incl), [], @nanmean);
        
        if ischar(S.plot_args)
            S.plot_args = {S.plot_args};
        end
        plot(conds, p_ch, S.plot_args{:});
        ylim([0, 1]);
        
        bml.plot.beautify;
        bml.plot.axis_margin;
        x_ticks = get(gca, 'XTick');
        set(gca, ...
            'XTick', [x_ticks(1), 0, x_ticks(end)], ...
            'YTick', [0, 0.5, 1]);
        xlabel('Mean evidence');
        ylabel('P_{ch}');
        
        % Outputs
        h = struct;
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