classdef RevCorBayesD1 ...
        < IxnKernel.EvTime.CommonWorkspace
    % Given a Tnd that is independent of the evidence, estimate P(t=Tnd).
    % (e.g., when Tnd is attached to the end of the Td)
    
%% Settings
properties
    % Ev = IxnKernel.EvTime.EvTimeD1; % Inherited from IxnKernel.EvTime.CommonWorkspace
    EP = IxnKernel.RevCor.D1.Ev2PchD1;
    EP_noTnd = IxnKernel.RevCor.D1.Ev2PchD1; % Ev2PchD1 w/ no Tnd
    
    n_boot = 100;
    
    S_Dtb_EP = varargin2S({
            ... % Arguments for SimDtbD1
            'drift', 0
            'sigmasq_ev', 1/2
            'sigmasq_internal', 1/2
            'bound', [-1, 1]
            'n_trial', 5e3 % 1e4
            'smooth_ev_args', {'normal', 0}
            'smooth_internal_args', {'normal', 0}
            ...
            ... % Arguments for SimDtbUtilD1
            't_util_spec', ...
                {'ig_p_sts'; 100; [30, 30, 70, 70]} % [40, 40, 60, 60]} % [100, 100, 0, 0]} % [75, 0, 75, 0]} % 
%                 {'ig_p_ens'; 100; [30, 30, 70, 70]} % [40, 40, 60, 60]} % [100, 100, 0, 0]} % [75, 0, 75, 0]} % 
%                 {'ig_en'; 200}
            ...
            ... % Arguments for Ev.get_S_filt_intervals,
            ... % used in Ev2PchD1, Dtb.get_ev_aligned and get_util_aligned
            'align', 'st' % 'rt' % 'st' % 
            'interval_width_ms', 100
            'interval_st_ms', 0
            'interval_ix', 1:6
            'interval_spacing_ms', 100
            'intervals_ms', []
            'rt_incl_ms', [0, 5000]
            });
end
%% Internal
properties
    Dtb = IxnKernel.RevCor.D1.SimDtbUtilD1; % Used for simulation
    Dtb_noTnd = IxnKernel.RevCor.D1.SimDtbUtilD1; % Used for simulation w/ no Tnd
end
%% Delegates
properties
    ch % (tr, 1)
    rt_fr % (tr, 1)
    ev_summary_interval % (tr, itv, 1)
    util % (tr, itv, 1)
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
    
    % Bootstrap results
    p_util_est_est % p_util_est_est(used+1, n_itv_passed+1)
    p_util_est_ci % p_util_est_ci(used+1, n_itv_passed+1, [lb, ub])
    p_util_est_boot % p_util_est_boot(used+1, n_itv_passed+1, boot)
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
        tic;
        %% Simulate
        [ch, ev_summary_interval, util, f_p_ch_given_util, p_util, S] = ...
            Rev.sim_ch_multi_t_bins(varargin{:});
        
        %% Estimate
        [p_util_est, p_util_est_by_n_used_llk] = ...
            Rev.ch2p_util_multi_t_bins_unit( ...
                ch, ev_summary_interval, f_p_ch_given_util);
        toc;
            
        %% Plot
        Rev.plot_p_util_all;
%         plot(p_util, p_util_est, 'o');
        
        %% Output
        res = packStruct(ch, ev_summary_interval, ...
            util, f_p_ch_given_util, p_util, ...
            p_util_est, p_util_est_by_n_used_llk, Rev);
        
        file = Rev.get_file;
        save(file, '-struct', 'res');
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
        S = Rev.get_S_Dtb_EP(varargin{:});
        C = S2C(S);
        
        %%
        Dtb = IxnKernel.RevCor.D1.SimDtbUtilD1(C{:});
        Rev.Dtb = Dtb;
        Dtb.simulate;
        Rev.Ev = Dtb.Ev;
        
        %%
        EP = IxnKernel.RevCor.D1.Ev2PchD1;
        Rev.EP = EP;
        EP.Ev = Dtb.Ev;
        EP.fit_intervals(C{:});
        
        %%
        tr_incl = Dtb.Ev.get_tr_incl(C{:});
        ch = Dtb.Ev.ch(tr_incl);
        ev_summary_interval = Dtb.get_ev_summary_intervals(C{:});
        
        %%
        util = Dtb.get_util_intervals(C{:});
        
        p_util = nanmean(util);
        Rev.p_util_sim = p_util;
        imagesc(util); % DEBUG
        
        %%
        C_noTnd = varargin2C({
            't_util_spec', {'all'}
            }, C);
        Dtb_noTnd = IxnKernel.RevCor.D1.SimDtbUtilD1(C_noTnd{:});
        Rev.Dtb_noTnd = Dtb_noTnd;
        
        EP_noTnd = IxnKernel.RevCor.D1.Ev2PchD1;
        Rev.EP_noTnd = EP_noTnd;
        Dtb_noTnd.simulate;
        EP_noTnd.Ev = Dtb_noTnd.Ev;
        EP_noTnd.fit_intervals(C{:});
        
        %%
        % p_ch(tr) = f_p_ch_given_util(ev(tr,itv), itv)
        f_p_ch_given_util = EP_noTnd.get_f_p_ch_given_util;
    end
    function S = get_S_Dtb_EP(Rev, varargin)
        % Settings used in demos        
        S = varargin2S(varargin, Rev.S_Dtb_EP);
    end
    function [p_util_est_est, p_util_est_ci, p_util_est_boot, ...
            p_util_est_by_n_used_lik, p_rest_by_n_used, p_util_jt] = ...
            ...
            ch2p_util_multi_t_bins_boot( ...
                Rev, ch, ev_summary_interval, f_p_ch_given_util)
        % Bootstrap estimate of the momentary evidence use.
        %
        % [p_util_est_est, p_util_est_ci, p_util_est_boot, ...
        %     p_util_est_by_n_used_lik, p_rest_by_n_used, p_util_jt] = ...
        %     ...
        %     ch2p_util_multi_t_bins_boot( ...
        %         Rev, ch, ev_summary_interval, f_p_ch_given_util)        
        %
        % ch(tr, 1) : 0 or 1
        % ev_summary_interval(tr, itv)
        % p_ch = f_p_ch_given_util(ev, n_passed+1)
        %
        % p_util_est_est(n_itv_passed+1, 1)
        % p_util_est_ci(n_itv_passed+1, [lb, ub])
        % p_util_est_boot(n_itv_passed+1, boot)
        % p_util_est_by_n_used_lik(n_passed+1, n_used+1)
        % p_rest_by_n_used(n_passed+1, n_used+1)
        % p_util_jt(used+1, n_passed+1, n_used+1)
            
        n_boot = Rev.n_boot;
        n_tr = size(ch, 1);
        n_itv = size(ev_summary_interval, 2);
        
        p_util_est_boot = zeros(n_itv, n_boot);
        
        t_st = tic;
        for i_boot = 1:n_boot
            if i_boot == 1
                % Use the original.
                ix = (1:n_tr)';
            else
                % Resample with replacement.
                ix = randi(n_tr, [n_tr, 1]);
            end
            ch1 = ch(ix,:);
            ev_summary_interval1 = ev_summary_interval(ix, :);
            
            [p_util_est1, p_util_est_by_n_used_lik1, ...
                p_rest_by_n_used1, p_util_jt1] = ...
                    ch2p_util_multi_t_bins_unit( ...
                        Rev, ch1, ev_summary_interval1, f_p_ch_given_util, ...
                        'to_cache', i_boot == 1);
                    
            if i_boot == 1
                p_util_est_by_n_used_lik = p_util_est_by_n_used_lik1;
                p_rest_by_n_used = p_rest_by_n_used1;
                p_util_jt = p_util_jt1;
            end
            
            p_util_est_boot(:,i_boot) = p_util_est1;
            
            t_el = toc(t_st);
            fprintf('Bootstrap #1-#%d took %1.1f s\n', i_boot, t_el);
        end
        p_util_est_est = median(p_util_est_boot, 2);
        p_util_est_ci = prctile(p_util_est_boot, [2.5, 97.5], 2);
        
        Rev.p_util_est_est = p_util_est_est;
        Rev.p_util_est_ci = p_util_est_ci;
        Rev.p_util_est_boot = p_util_est_boot;
    end
    function [p_util_est, p_util_est_by_n_used_lik, ...
                p_rest_by_n_used, p_util_jt] = ...
            ch2p_util_multi_t_bins_unit( ...
                Rev, ch, ev_summary_interval, f_p_ch_given_util, varargin)
        % [p_util_est, p_util_est_by_n_est] = ch2p_util_multi_t_bins_unit( ...
        %         Rev, ch, ev_summary_interval, f_p_ch_given_util, ...)
        %
        % p_ch = f_p_ch_given_util(ev, n_passed+1)
        % p_util_est(1, n_itv_passed+1)
        % p_util_est_by_n_used(n_passed+1, n_used+1)
        % p_rest_by_n_used(n_passed+1, n_used+1)
        % p_util_jt(n_passed+1, n_used+1)
        %
        % OPTIONS:
        % 'to_cache', true
        
        S = varargin2S(varargin, {
            'to_cache', true
            });
        
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
        if S.to_cache
            Rev.p_util_est_by_n_used_lik = p_util_est_by_n_used_lik;
            Rev.p_rest_by_n_used = p_rest_by_n_used;
            Rev.p_util_jt = p_util_jt;
            Rev.p_util_est = p_util_est;
        end
%         imagesc(p_util_est_by_n_used_lik);
    end
    function plot_p_util_all(Rev)
        colormap(parula);
        
        nR = 2;
        nC = 2;
        for ii = {
                {1,1}, 'p_util_est_by_n_used_lik', '$p_{acq|a,s}$'
                {1,2}, 'p_rest_by_n_used', '$p_{a|s}$'
                {2,1}, 'p_util_jt', '$p_{acq|a,s} \circ p_{a|s} = p_{acq,a|s}$'
%                 {1,1}, 'p_util_est_by_n_used_lik', '$\hat{p}_{u,\tau,\upsilon}$'
%                 {1,2}, 'p_rest_by_n_used', '$P_{\tau,\upsilon}$'
%                 {2,1}, 'p_util_jt', '$\hat{p}_{u,\tau,\upsilon} \circ P_{\tau,\upsilon}$'
                }'
            [rc, name, str_title] = deal(ii{:});
            ax(rc{:}) = subplotRC(nR, nC, rc{:});
            
            v = Rev.(name);
            n = size(v, 1);
            
            imagesc(1:n, 1:n, v');
%             imagesc(0:(n-1), 0:(n-1), v);
            axis square;
            set(gca, 'CLim', [0 1]); % , 'TickDir', 'out');
            h_col = colorbar;
            if ~isequal(rc, {1,2})
                set(h_col, 'Visible', 'off');
            else
                set(h_col, 'TickDir', 'out');
            end
            set(gca, 'XTick', [], 'YTick', []);
            
            title(str_title, 'Interpreter', 'Latex'); % strrep(name, '_', '-'));
            if isequal(rc, {2,1})
                xlabel('# Intervals Shown (s)');
                ylabel({'# Intervals', 'Acquired', '(a)'}, ...
                    'Rotation', 0, ...
                    'HorizontalAlignment', 'right', ...
                    'VerticalAlignment', 'middle');
%                 ylabel('# Intervals Acquired (a)');
%                 xlabel('# Intervals used (\upsilon)');
%                 ylabel('# Intervals passed (\tau)');
            end
        end
        
        ax(2,2) = subplotRC(2,2,2,2);
        
        np = numel(Rev.p_util_sim);
        ix_itv = 1:np;
%         ix_itv = 0:(np-1);
        
        x_st = [0.5, ix_itv + 0.5];
        y_st = [Rev.p_util_sim, Rev.p_util_sim(end)];
        stairs(x_st, y_st, 'k-');
%         plot(ix_itv, Rev.p_util_sim, 'k-');
        hold on;
        plot(ix_itv, Rev.p_util_est, 'ko');
        hold off;
        axis square;
        
        set(gca, 'XTick', [1, 6]);
        ylim([-.05, 1.05]);
        ylabel('$p_{acq|s}$', ...
            'Interpreter', 'Latex', ...
            'Rotation', 0, ...
            'HorizontalAlignment', 'right', ...
            'VerticalAlignment', 'middle');
%         ylabel('$p_{u,\tau}$', 'Interpreter', 'Latex');
%         ylabel('P_{utilized}');
        xlabel('# Intervals Shown (s)');
%         title('p-util-est');
%         set(gca, 'Xtick', 1:2:10);
        xlim([0.5, np + 0.5]);
        bml.plot.beautify(gca);
        
        %%
        bml.plot.position_subplots(ax, ...
            'margin_left', 0.13, ...
            'margin_bottom', 0.15, ...
            'margin_right', 0.075, ...
            'margin_top', 0.075, ...
            'btw_row', 0.15, ...
            'btw_col', 0.03);
        
        %%
        file = Rev.get_file({'plt', 'p_util_all'});
        savefigs(file, 'size', [400, 300], ...
            'ext', {'.png', '.tif', '.fig'});
    end
    function [h, he, h_orig] = plot_p_util_boot(Rev)
        %%        
        p = Rev.p_util_est_est;
        ci = Rev.p_util_est_ci;
        e = bsxfun(@minus, ci, p);

        n = numel(p);
        itv = 0:(n-1);

        [h, he] = errorbar_wo_tick(itv, p, e(:,1), e(:,2), {
                'Marker', 'o'
                'MarkerEdgeColor', 'w'
                'MarkerFaceColor', 'k'
                });
            
        if ~isempty(Rev.p_util_sim)
            p_sim = Rev.p_util_sim(:);
            h_orig = plot(itv, p_sim, 'k-', 'LineWidth', 2);
        end
        
        ylabel('P_{use}');
        xlabel('Interval (\tau)');
        bml.plot.beautify;
        
        xlim([-0.5, n - 0.5]);
        ylim([-0.05, 1.05]);
        set(gca, 'YTick', 0:0.25:1, 'YTickLabel', {'0', '', '', '', '1'});
              
        %%
        file = Rev.get_file({'plt', 'p_use'});
        savefigs(file);
    end  
    function [h, h_est] = plot_p_util_boot_violin(Rev)
        % [h, h_est] = plot_p_util_boot_violin(Rev)
        
        %%
        hs = distributionPlot(Rev.p_util_est_boot');
        h = hs{1};
        h_est = hs{2};
        legend(h_est, {'Mean', 'Median'}, ...
            'Location', 'NorthEastOutside');
        
        n_itv = size(Rev.p_util_est_boot, 1);
        set(gca, ...
            'XTickLabel', (1:n_itv) - 1);
        xlim([0.5, n_itv + 0.5]);
        ylim([-0.01, 1.01]);
        ylabel('P_{use}');
        xlabel('Interval (\tau)');
        bml.plot.beautify;
        
        %%
        file = Rev.get_file({'plt', 'p_use_vln'});
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
        xlabel('True P_{use}');
        ylabel('Estimated P_{use}');
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
%% Plotting / Saving
methods
    function fs = get_file_fields0(~)
        fs = {
            'n_boot', 'nbt'
            };
    end
    function plot_n_save_sim(Rev)
        Rev.plot_sim;
        Rev.save_mat;
    end
    function plot_sim(Rev)
        %% Plot
%         Rev.plot_p_util_boot; 
        Rev.plot_p_util_boot_violin; 
    end
    function save_mat(Rev)        
        %% Output
        props_to_save = {
            'n_boot'
            'ch'
            'rt_fr'
            'ev_summary_interval'
            'util'
            'p_util_est_by_n_used_lik'
            'p_rest_by_n_used'
            'p_util_jt'
            'p_util_sim'
            'p_util_est'
            'p_util_est_est'
            'p_util_est_ci'
            'p_util_est_boot'
            'S_file'
            };
        res = copyFields(struct, Rev, props_to_save);
        res.Rev = Rev; % Takes 2x space, but for convenience...
        
        %% Save
        file = Rev.get_file;
        save(file, '-struct', 'res');        
        fprintf('Saved Rev.%s \n to %s\n', ...
            sprintf('%s, ', props_to_save{:}), ...
            file);
    end
end
%% Importing
methods
    function batch_fit_aft_import(Rev, files, varargin)
        n = numel(files);
        for ii = 1:n
            file1 = files{ii};
            Rev.fit_aft_import(file1, varargin{:});
        end
    end
    function fit_aft_import(Rev, file, varargin)
        % fit_aft_import(Rev, file, varargin)
        %
        % file: 1D file to provide no_Tnd fit as well as the data.
        %
        % Options: 
        % Any options as in one of the following:
        % - Rev.get_S_Dtb_EP()
        % - Rev.import_pdf
        %   - dif_rel_incl for EvTimeD1Pdf.import_pdf
        %   - dif_incl for EvTimeD2.import_data
        
        %% Default files
        if ~exist('file', 'var')
            file = ...
                '../Data_2D/Fit.D1.BndEn2Bnd.Main/sbj=DX+prd=RT+tsk=A+dtk=2+dmr=1+trm=201+eor=t+dft=C+bnd=A2+ssq=C+tnd=i+ntnd=2+msf=1+fsqs=1+frst=10+fbst=0+exp=pred.mat';
%                 '../Data_2D/Fit.D1.BoundedEn.Main/sbj=DX+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+dft=C+bnd=A2+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+frst=10+fbst=0+exp=pred.mat';
%                 '../Data_2D/Fit.D1.BoundedEn.Main/sbj=DX+prd=RT+tsk=H+dtk=1+dmr=1+trm=201+eor=t+bnd=C+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=1+fbst=1+exp=pred.mat';
        end
        
        %% Import
        [ch, ev_summary_interval, f_p_ch_given_util] = ...
            Rev.import_pdf(file, varargin{:});
        
        %% Estimate
%         [p_util_est, p_util_est_by_n_used_lik, ...
%                 p_rest_by_n_used, p_util_jt] = ...
%                 Rev.ch2p_util_multi_t_bins_unit( ...
%                     ch, ev_summary_interval, f_p_ch_given_util);
                
        [p_util_est_est, p_util_est_ci, p_util_est_boot, ...
            p_util_est_by_n_used_lik, p_rest_by_n_used, p_util_jt] = ...
                Rev.ch2p_util_multi_t_bins_boot( ...
                    ch, ev_summary_interval, f_p_ch_given_util);
        
        %% Plot
        clf;
        Rev.plot_sim;
        
        %% Save
        Rev.save_mat;
    end
    function [ch, ev_summary_interval, f_p_ch_given_util] = ...
            import_pdf(Rev, file, varargin)
        % Import RT_data and Td_pred to Dtb and EP.
        % Specifically, RT_data_pdf_tr goes to Dtb and EP,
        % and Td_pred_pdf_tr goes to Dtb_noTnd and EP_noTnd.
        %
        % import_pdf(Rev, file)
        %
        % file: 1D file to provide no_Tnd fit as well as the data.

        %%
        Rev.Dtb = IxnKernel.RevCor.D1.SimDtbUtilD1;
        Rev.Dtb.Ev = IxnKernel.EvTime.EvTimeD1Pdf;
        Rev.Dtb.Ev.import_pdf(file, ...
            'src', 'RT_data', varargin{:}); % 1D data
        
        %%
        Rev.Dtb_noTnd = IxnKernel.RevCor.D1.SimDtbUtilD1;
        Rev.Dtb_noTnd.Ev = IxnKernel.EvTime.EvTimeD1Pdf;
        Rev.Dtb_noTnd.Ev.import_pdf(file, ...
            'src', 'Td_pred', varargin{:}); % 1D fit
        
        %%
        Rev.Ev = Rev.Dtb.Ev;
        Rev.ch = Rev.Dtb.Ev.ch;
        Rev.rt_fr = Rev.Dtb.Ev.rt_fr;
        Rev.ev_summary_interval = Rev.Dtb.get_ev_summary_intervals;
        Rev.util = []; % Unknown
        
        Rev.EP = IxnKernel.RevCor.D1.Ev2PchD1;
        Rev.EP.Ev = Rev.Dtb.Ev;
        
        Rev.EP_noTnd = IxnKernel.RevCor.D1.Ev2PchD1;
        Rev.EP_noTnd.Ev = Rev.Dtb_noTnd.Ev;
        
        %%
        C = S2C(Rev.get_S_Dtb_EP(varargin{:}));
        Rev.EP.fit_intervals(C{:}, 'ch_from_pdf', false);
        Rev.EP_noTnd.fit_intervals(C{:}, 'ch_from_pdf', false); % true);
        
        %%
        Rev.S_file_ = copyFields(Rev.S_file_, Rev.Dtb.Ev.S_file);
        
        %% Outputs        
        f_p_ch_given_util = Rev.EP_noTnd.get_f_p_ch_given_util;

        Dtb = Rev.Dtb;
        C = S2C(Rev.get_S_Dtb_EP(varargin{:}));
        tr_incl = Dtb.Ev.get_tr_incl(C{:});
        ch = Dtb.ch(tr_incl,:);
        ev_summary_interval = Dtb.get_ev_summary_intervals(C{:});
        
        %% Copy variables to save
        Rev.ch = Rev.Dtb.ch;
        Rev.rt_fr = Rev.Dtb.Ev.rt_fr;
        Rev.ev_summary_interval = ev_summary_interval;        
    end
end
end