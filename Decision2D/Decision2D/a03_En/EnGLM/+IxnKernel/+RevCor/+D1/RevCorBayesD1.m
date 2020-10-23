classdef RevCorBayesD1 ...
        < IxnKernel.EvTime.CommonWorkspace
    % Given a Tnd that is independent of the evidence, estimate P(t=Tnd).
    % (e.g., when Tnd is attached to the end of the Td)
    
%% Settings
properties
    % Ev = IxnKernel.EvTime.EvTimeD1; % Inherited from IxnKernel.EvTime.CommonWorkspace
    EP = IxnKernel.RevCor.D1.Ev2PchD1;
    
    EP_noTnd = IxnKernel.RevCor.D1.Ev2PchD1; % Ev2PchD1 w/ no Tnd
end
%% Internal
properties
    Sim = IxnKernel.RevCor.D1.SimDtbUtilD1; % Used for simulation
    Sim_noTnd = IxnKernel.RevCor.D1.SimDtbUtilD1; % Used for simulation w/ no Tnd
end
%% Results
properties    
    % p_util_est_by_n_used_lik(n_itv_passed+1, n_itv_used+1)
    % : estimated marginal probability of utilization across n_itv_used.
    p_util_est_by_n_used_lik

    % p_rest_by_n_used(n_itv_passed+1, n_itv_used+1)
    % : estimated marginal probability of utilization across n_itv_used.
    p_rest_by_n_used

    % p_util_jt(n_itv_passed+1, n_itv_used+1)
    % : estimated marginal probability of utilization across n_itv_used.
    p_util_jt

    % p_util_sim(n_itv_passed+1, 1)
    % : p_util from the true util() matrix used in simulation.
    p_util_sim
    
    % p_util_est(n_itv_passed+1, 1)
    % : estimated marginal probability of utilization across n_itv_used.
    p_util_est    
end
%% Init
methods
    function Rev = RevCorBayesD1(varargin)
        if nargin > 0
            Rev.init(varargin{:});
        end
    end
    function init(Rev, varargin)
        varargin2props(Rev, varargin, true);        
    end
end
%% Main
methods
    function main(Rev, varargin)
        Rev.demo_ch2p_util_multi_t_bins_unit(varargin{:});
    end
end
%% Demo with multiple time bins
methods
    function [res, S] = demo_ch2p_util_multi_t_bins_unit(Rev, varargin)
        %% Simulate
        [ch, ev_summary_interval, util, f_p_ch_given_util, p_util, S] = ...
            Rev.sim_ch_multi_t_bins(varargin{:});
        
        %% Estimate
        [p_util_est, p_util_est_by_n_used_llk] = ...
            Rev.ch2p_util_multi_t_bins_unit( ...
                ch, ev_summary_interval, f_p_ch_given_util);
        
        %% Plot
        Rev.plot_p_util_all;
%         plot(p_util, p_util_est, 'o');
        
        %% Output
        res = packStruct(ch, ev_summary_interval, ...
            util, f_p_ch_given_util, p_util, ...
            p_util_est, p_util_est_by_n_used_llk);
    end
    function [ch, ev_summary_interval, util, f_p_ch_given_util, p_util, S] = ...
            sim_ch_multi_t_bins(Rev, varargin)
        % [ch, util, p_ch_given_util, S] = ...
        %     sim_ch_multi_t_bins(Rev, p_util, varargin)
        %
        % p_util(1, t_bin)
        % n_tr : scalar
        %
        % ch(tr, 1)
        % util(tr, t_bin)
        % p_ch_given_util(tr, util+1, t_bin_used+1)
        
        %%
        S = varargin2S(varargin, {
            ... % Arguments for SimDtbD1
            'drift', 0
            'sigmasq_ev', 1/2
            'sigmasq_internal', 1/2
            'bound', [-1, 1]
            'n_trial', 1e4
            'smooth_ev_ms', 0
            'smooth_interval_ms', 0
            ... % Arguments for SimDtbUtilD1
            't_util_spec', ...
                {'ig_p_ens'; 100; [30, 30, 70, 70]} % [40, 40, 60, 60]} % [100, 100, 0, 0]} % [75, 0, 75, 0]} % 
%                 {'ig_en'; 200}
            ... % Arguments for Ev.get_S_filt_intervals,
            ... % used in Ev2PchD1, Sim.get_ev_aligned and get_util_aligned
            'align', 'rt'
            'interval_width_ms', 100
            'interval_st_ms', 0
            'interval_en_ms', 1000
            'interval_spacing_ms', 100
            'intervals_ms', []
            'rt_incl_ms', [1100, 5000]
            });
        
        C = S2C(S);
        
        %%
        Sim = IxnKernel.RevCor.D1.SimDtbUtilD1(C{:});
        Rev.Sim = Sim;
        Sim.simulate;
        Rev.Ev = Sim.Ev;
        
        %%
        EP = IxnKernel.RevCor.D1.Ev2PchD1;
        Rev.EP = EP;
        EP.Ev = Sim.Ev;
        EP.fit_intervals(C{:});
        
        %%
        tr_incl = Sim.Ev.get_tr_incl(C{:});
        ch = Sim.Ev.ch(tr_incl);
        ev_summary_interval = Sim.get_ev_summary_intervals(C{:});
        
        %%
        util = Sim.get_util_intervals(C{:});
        
        p_util = nanmean(util);
        Rev.p_util_sim = p_util;
        imagesc(util); % DEBUG
        
        %%
        C_noTnd = varargin2C({
            't_util_spec', {'all'}
            }, C);
        Sim_noTnd = IxnKernel.RevCor.D1.SimDtbUtilD1(C_noTnd{:});
        Rev.Sim_noTnd = Sim_noTnd;
        
        EP_noTnd = IxnKernel.RevCor.D1.Ev2PchD1;
        Rev.EP_noTnd = EP_noTnd;
        Sim_noTnd.simulate;
        EP_noTnd.Ev = Sim_noTnd.Ev;
        EP_noTnd.fit_intervals(C{:});
        
        %%
        % p_ch(tr) = f_p_ch_given_util(ev(tr,itv), itv)
        f_p_ch_given_util = EP_noTnd.get_f_p_ch_given_util;
    end
    function [p_util_est, p_util_est_by_n_used_lik, ...
                p_rest_by_n_used, p_util_jt] = ...
            ch2p_util_multi_t_bins_unit( ...
                Rev, ch, ev_summary_interval, f_p_ch_given_util)
        % [p_util_est, p_util_est_by_n_est] = ch2p_util_multi_t_bins_unit( ...
        %         Rev, ch, ev_summary_interval, f_p_ch_given_util)
        %
        % p_ch = f_p_ch_given_util(ev, n_passed+1)
        % p_util_est(1, n_itv_passed+1)
        % p_util_est_by_n_used(n_passed+1, n_used+1)
        % p_rest_by_n_used(n_passed+1, n_used+1)
        % p_util_jt(n_passed+1, n_used+1)
        
        n_itv = size(ev_summary_interval, 2);
        p_util_est_by_n_used_lik = zeros(n_itv, n_itv);
        
        %% Estimate likelihood
        for n_passed = 0:(n_itv - 1)
            for n_used = 0:n_passed
                % p_ch_given_util(tr,1)
                p_ch_given_util = f_p_ch_given_util( ...
                    ev_summary_interval(:, n_passed + 1), n_used);
                
                p_util_est_by_n_used_lik(n_passed + 1, n_used + 1) = ...
                    Rev.ch2p_util_unit( ...
                        ch, p_ch_given_util);
            end
        end
        
        %% Prob left by n_passed using recurrence relation
        p_rest_by_n_used = zeros(n_itv, n_itv);
        p_rest_by_n_used(1,1) = 1;
        for n_passed = 0:(n_itv - 2)
            for n_used = 0:n_passed
                p_rest1 = p_rest_by_n_used( ...
                    n_passed + 1, n_used + 1);
                p_util1 = p_util_est_by_n_used_lik( ...
                    n_passed + 1, n_used + 1);
                
                % If unused, add to the cell in the south.
                p_rest_by_n_used(n_passed + 2, n_used + 1) = ...
                    p_rest_by_n_used(n_passed + 2, n_used + 1) ...
                    + p_rest1 .* (1 - p_util1);
                
                % If used, add to the cell in the southeast.
                p_rest_by_n_used(n_passed + 2, n_used + 2) = ...
                    p_rest_by_n_used(n_passed + 2, n_used + 2) ...
                    + p_rest1 .* p_util1;
            end
        end
        
        p_util_jt = p_util_est_by_n_used_lik .* p_rest_by_n_used;
        p_util_est = sum(p_util_jt, 2)';
        
        %% Cache for plotting
        Rev.p_util_est_by_n_used_lik = p_util_est_by_n_used_lik;
        Rev.p_rest_by_n_used = p_rest_by_n_used;
        Rev.p_util_jt = p_util_jt;
        Rev.p_util_est = p_util_est;
%         imagesc(p_util_est_by_n_used_lik);
    end
    function plot_p_util_all(Rev)
        nR = 2;
        nC = 2;
        for ii = {
                {1,1}, 'p_util_est_by_n_used_lik', '$\hat{p}_{u,\tau,\upsilon}$'
                {1,2}, 'p_rest_by_n_used', '$P_{\tau,\upsilon}$'
                {2,1}, 'p_util_jt', '$\hat{p}_{u,\tau,\upsilon} \circ P_{\tau,\upsilon}$'
                }'
            [rc, name, str_title] = deal(ii{:});
            ax(rc{:}) = subplotRC(nR, nC, rc{:});
            
            v = Rev.(name);
            n = size(v, 1);
            
            imagesc(0:(n-1), 0:(n-1), v);
            axis square;
            h_col = colorbar;
            
            title(str_title, 'Interpreter', 'Latex'); % strrep(name, '_', '-'));
            if isequal(rc, {2,1})
                xlabel('# Intervals used (\upsilon)');
                ylabel('# Intervals passed (\tau)');
            end
        end
        
        ax(2,2) = subplotRC(2,2,2,2);
        
        np = numel(Rev.p_util_sim);
        ix_itv = 0:(np-1);
        
        plot(ix_itv, Rev.p_util_sim, 'k-');
        hold on;
        plot(ix_itv, Rev.p_util_est, 'ko');
        hold off;
        
        ylim([-.05, 1.05]);
        ylabel('$p_{u,\tau}$', 'Interpreter', 'Latex');
%         ylabel('P_{utilized}');
        xlabel('Interval (\tau)');
%         title('p-util-est');
%         set(gca, 'Xtick', 1:2:10);
        xlim([-1, np]);
        bml.plot.beautify;
        
        %%
        bml.plot.position_subplots(ax, ...
            'margin_left', 0.05, ...
            'margin_bottom', 0.15, ...
            'margin_right', 0.075, ...
            'margin_top', 0.075, ...
            'btw_row', 0.15, ...
            'btw_col', 0.2);
        
        %%
        file = Rev.get_file({'plt', 'p_util_all'});
        savefigs(file);
    end
end
%% Demo with one time bin
methods
    function demo_ch2p_util_batch(Rev, varargin)
        %%
        n_tr_per_cond = 1e4;
        p_ch_given_util = 0.7;
        [res, S] = Rev.demo_ch2p_util_unit( ...
            'p_ch_given_util', [
                repmat([0.5, p_ch_given_util], [n_tr_per_cond, 1])
                repmat([0.5, p_ch_given_util], [n_tr_per_cond, 1])
                repmat([0.5, p_ch_given_util], [n_tr_per_cond, 1])
                ], ...
            'p_util', [
                repmat(0.2, [n_tr_per_cond, 1])
                repmat(0.5, [n_tr_per_cond, 1])
                repmat(0.8, [n_tr_per_cond, 1])
                ]);
        plot_demo_ch2p_util_unit(Rev, S.p_util, res.p_util_est)
    end
    function [res, S] = demo_ch2p_util_unit(Rev, varargin)
        S = varargin2S(varargin, {
            'p_ch_given_util', []
            'p_util', []
            'n_tr', []
            });
        
        %%
        [ch, util, cond, S] = Rev.sim_ch_given_p_util_and_p_ch_given_util( ...
            S.p_ch_given_util, S.p_util, S.n_tr);
        [p_util_est, p_util_ests] = Rev.ch2p_util( ...
            ch, S.p_ch_given_util, cond);
        
        res = packStruct(p_util_est, p_util_ests, ch, util, cond);
    end
    function plot_demo_ch2p_util_unit(Rev, p_util, p_util_est)
        plot(p_util, p_util_est, 'ko');
        bml.plot.beautify;
        crossLine('NE', 0, {'--', [0 0 0] + 0.7});
        xlabel('True P_{utilized}');
        ylabel('Estimated P_{utilized}');
    end
    function [ch, util, cond, S] = sim_ch_given_p_util_and_p_ch_given_util(~, ...
            p_ch_given_util, p_util, n_tr)
        % [ch, util, cond] = sim_ch_given_p_util_and_p_ch_given_util(Rev, ...
        %             p_ch_given_util, p_util, n_tr)
        %
        % p_ch_given_util(tr, util+1) = P(ch(tr) = 1 | util)
        % p_util(tr, 1) = P(util(tr) = 1)
        % n_tr : scalar.
        %
        % ch(tr, 1) : 0 or 1.
        % util(tr, 1) : 0 or 1.
        % cond(tr, 1) : 1..numel(unique(util))
        if nargin < 2 || isempty(p_ch_given_util)
            p_ch_given_util = [0.5, 0.8];
        else
            if size(p_ch_given_util, 2) == 1
                p_ch_given_util = ...
                    [0.5 + zeros(size(p_ch_given_util, 1), 1), ...
                     p_ch_given_util];
            end
            assert(size(p_ch_given_util, 2) == 2);
        end
        if nargin < 3 || isempty(p_util)
            p_util = 0.8;
        end
        if nargin < 4 || isempty(n_tr)
            if size(p_ch_given_util, 1) > 1
                n_tr = size(p_ch_given_util, 1);
            elseif size(p_util, 1) > 1
                n_tr = size(p_util, 1);
            else
                n_tr = 1e5;
            end
        end
        
        if size(p_ch_given_util, 1) == 1
            p_ch_given_util = repmat(p_ch_given_util, [n_tr, 1]);
        else
            assert(size(p_ch_given_util, 1) == n_tr);
        end
        if size(p_util, 1) == 1
            p_util = repmat(p_util, [n_tr, 1]);
        else
            assert(size(p_util, 1) == n_tr);
        end
        
        [~,~,cond] = unique(p_util);
        
        %% Simulate ch & util
        util = rand(n_tr, 1) < p_util;
        ch = false(n_tr, 1);
        
        for util1 = 0:1
            incl = util == util1;
            ch(incl) = rand([nnz(incl), 1]) ...
                < p_ch_given_util(incl, util1 + 1);
        end
        
        %% Return settings
        if nargout >= 4
            S = packStruct(p_ch_given_util, p_util, n_tr);
        end
    end
    function [p_util_est, p_util_ests] = ch2p_util(Rev, ...
                    ch, p_ch_given_util, cond)
        % [p_util_est, p_util_ests] = ch2p_util(Rev, ...
        %                     ch, p_ch_given_util, cond)                
        %
        % ch(tr, 1) : 0 or 1.
        % p_ch_given_util(tr, util+1) = P(ch(tr) = 1 | util)
        % cond(tr, 1) : categorical variable. 
        %               p_util_est is estimated per each unique condition.
        %
        % p_util_est(tr, 1) : p_util for cond(tr).
        % p_util_ests(1, d_cond) : p_util for conds(d_cond).
        
        assert(iscolumn(ch));
        assert(iscolumn(cond));
        n_tr = size(cond, 1);
        
        assert(size(p_ch_given_util, 1) == n_tr);
        assert(size(p_ch_given_util, 2) == 2);
        
        [~, ~, d_cond] = unique(cond);
        n_conds = max(d_cond);
        p_util_ests = zeros(1, n_conds);
        p_util_est = zeros(n_tr, 1);
        for i_cond = 1:n_conds
            incl = d_cond == i_cond;
            p_util_ests(i_cond) = Rev.ch2p_util_unit( ...
                ch(incl), p_ch_given_util(incl, :));
            p_util_est(incl) = p_util_ests(i_cond);
        end
    end
    function p_util_est = ch2p_util_unit(Rev, ch, p_ch_given_util)
        % p_ch_given_util(util+1) = P(ch = 1 | util)
        % p_ch_given_util(tr, util+1) = P(ch(tr) = 1 | util)
        %
        % p_util_est : scalar.

        if size(p_ch_given_util, 2) == 1
            n_tr = size(p_ch_given_util, 1);
            p_ch_given_util = [0.5 + zeros(n_tr, 1), p_ch_given_util];
        else
            assert(size(p_ch_given_util, 2) == 2);
        end
        
        %% Cost function
        [ch, p_ch_given_util] = bml.matrix.rows_w_no_nan(ch, p_ch_given_util);
        
        f = @(pu) -Rev.get_log_p_p_util_given_ch( ...
            pu, ch, p_ch_given_util);
%         disp(f(0.5)); % Test run
        
        %% Estimate p_util given p_ch_util and ch
        epsilon = 1e-6;
        opt = optimoptions('fmincon', 'Display', 'notify');
        [p_util_est, fval, exitflag, output, lambda, grad, hessian] = ...
            fmincon(f, ...
            0.5, [], [], [], [], ...
            epsilon, 1-epsilon, [], opt);
        
%         % DEBUG
%         x = 0.1:0.01:0.9;
%         plot(x, f(x));
%         disp(p_util_est);
    end
    function log_p_p_util = get_log_p_p_util_given_ch(~, ...
            p_util, ch, p_ch_given_util)
        % log_p_p_util = get_log_p_p_util_given_ch(Rev, ...
        %             p_util, ch, p_ch_given_util)        
        %
        % p_util : scalar
        % ch(tr,1) : 0 or 1
        % p_ch_given_util(tr, util1+1) = P(ch(tr)=+1 | util(tr)=util1)
        %
        % log_p_p_util : scalar
        
        epsilon = 1e-6;
        p_ch_given_util = min(max(p_ch_given_util, epsilon), 1 - epsilon);
        p_util = min(max(p_util, epsilon), 1 - epsilon);
        
        n_tr = size(ch, 1);
        p_ch_obs_given_util = zeros(n_tr, 2);
        p_ch_obs_given_util(ch,:) = p_ch_given_util(ch,:);
        p_ch_obs_given_util(~ch,:) = 1 - p_ch_given_util(~ch,:);
        
        log_p_p_util = sum( ...
            log(p_ch_obs_given_util(:,1) .* (1 - p_util) ...
              + p_ch_obs_given_util(:,2) .* p_util));
    end
end
end