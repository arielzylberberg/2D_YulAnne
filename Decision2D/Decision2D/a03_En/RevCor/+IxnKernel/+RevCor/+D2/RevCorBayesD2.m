classdef RevCorBayesD2 ...
        < IxnKernel.EvTime.CommonWorkspace
    % Given a Tnd that is independent of the evidence, estimate P(t=Tnd).
    % (e.g., when Tnd is attached to the end of the Td)
    
%% Settings
properties
    EP = IxnKernel.RevCor.D2.Ev2PchD2;
    EP_noTnd = IxnKernel.RevCor.D2.Ev2PchD2; % Ev2PchD2 w/ no Tnd
    
    n_boot = 100; % 100; % Increase to 200
    
    S_Dtb_EP = varargin2S({
        ... % Arguments for SimDtbD1
        'drift', 0
        'sigmasq_ev', 1/2
        'sigmasq_internal', 1/2
        'bound', [-1, 1]
        'n_trial', 1e4 % 1.5e3; % 5e3; % 2e3 % 2e2 % 4e2
        'smooth_ev_args', {'normal', 0} % 25}
        'smooth_internal_args', {'normal', 0}
        ...
        ... % Arguments for SimDtbUtilD1
        't_util_spec', {
            'itv_bmc'
            1
            [50, 50, 0]
            2
            [0, 50, 50]
            3
            [50, 0, 0]
            4
            [50, 0, 50]
            }
%         't_util_spec', ...
%             {'jt_st_sym'; 100; [100, 100, 0, 0]} % [0, 0, 100, 100]} % 
        ...
        ... % Arguments for Ev.get_S_filt_intervals,
        ... % used in Ev2PchD1, Dtb.get_ev_aligned and get_util_aligned
        'align', 'st'
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
    Dtb = IxnKernel.RevCor.D2.SimDtbUtilD2; % Used for simulation
    Dtb_noTnd = IxnKernel.RevCor.D2.SimDtbUtilD2; % Used for simulation w/ no Tnd
end
%% Delegates
properties
    ch % (tr, dim)
    rt_fr % (tr, 1)
    ev_summary_interval % (tr, itv, dim)
    util % (tr, itv, dim)
end
%% Results
properties    
    % p_util_est_by_n_used_lik(used1+1, used2+1, n_passed+1, n_used1+1, n_used2+1)
    % : estimated marginal momentary evidence use across n_itv_used.
    p_util_est_by_n_used_lik

    % p_rest_by_n_used(n_passed+1, n_used1+1, n_used2+1)
    % : estimated marginal momentary evidence use across n_itv_used.
    p_rest_by_n_used

    % p_util_jt(used1+1, used2+1, n_passed+1, n_used1+1, n_used2+1)
    % : estimated marginal momentary evidence use across n_itv_used.
    p_util_jt

    % p_util_est(used1+1, used2+1, n_itv_passed+1)
    % : p_util from the true util() matrix used in simulation.
    p_util_sim
    
    % p_util_est(used1+1, used2+1, n_itv_passed+1)
    % : estimated marginal momentary evidence use across n_itv_used.
    p_util_est
    
    % Bootstrap results
    p_util_est_est % p_util_est_est(used1+1, used2+1, n_itv_passed+1)
    p_util_est_ci % p_util_est_ci(used1+1, used2+1, n_itv_passed+1, [lb, ub])
    p_util_est_boot % p_util_est_boot(used1+1, used2+1, n_itv_passed+1, boot)
end
%% Init
methods
    function Rev = RevCorBayesD2(varargin)
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
        Rev.main_fit(varargin{:});
    end
    function main_demo(Rev, varargin)
        Rev.demo_ch2p_util_multi_t_bins_unit(varargin{:});
    end
end
%% Demo with multiple time bins
methods
    function [res, S] = demo_ch2p_util_multi_t_bins_boot(Rev, varargin)
        %% Simulate
        [ch, ev_summary_interval, util, f_p_ch_given_util, p_util, S] = ...
            Rev.sim_ch_multi_t_bins(varargin{:});
        
        %% Copy variables to save
        Rev.ch = Rev.Dtb.ch;
        Rev.rt_fr = Rev.Dtb.Ev.rt_fr;
        Rev.ev_summary_interval = ev_summary_interval;
        Rev.util = util;
                
        %% Bootstrap
        [p_util_est_est, p_util_est_ci, p_util_est_boot, ...
            p_util_est_by_n_used_lik, p_rest_by_n_used, p_util_jt] = ...
                Rev.ch2p_util_multi_t_bins_boot( ...
                    ch, ev_summary_interval, f_p_ch_given_util);
        
        %% Plot & Save
        Rev.plot_n_save_sim;
    end
    function [res, S] = demo_ch2p_util_multi_t_bins_unit(Rev, varargin)
        %% Simulate
        [ch, ev_summary_interval, util, f_p_ch_given_util, p_util, S] = ...
            Rev.sim_ch_multi_t_bins(varargin{:});
        
        %% Estimate
        tic;
        [p_util_est, p_util_est_by_n_used_llk] = Rev.ch2p_util_multi_t_bins_unit( ...
            ch, ev_summary_interval, f_p_ch_given_util);
        toc;
        
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
        % Get Rev.Dtb, EP, Dtb_noTnd, EP_noTnd, and ultimately
        % ch, ev_summary_interval, and util from Dtb.Ev, and
        % f_p_ch_given_util from EP_noTnd.
        %
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
        S = Rev.get_S_Dtb_EP;
        C = S2C(S);
        
        %%
        Dtb = IxnKernel.RevCor.D2.SimDtbUtilD2(C{:});
        Rev.Dtb = Dtb;
        Rev.add_children_props({'Dtb'});
        
        %%
        Dtb.simulate;
        Rev.Ev = Dtb.Ev;
        
        %%
        tic;
        EP = IxnKernel.RevCor.D2.Ev2PchD2;
        Rev.EP = EP;
        EP.Ev = Dtb.Ev;
        EP.fit_intervals(C{:});
        toc;
        
        %%
        tr_incl = Dtb.Ev.get_tr_incl(C{:});
        ch = Dtb.ch(tr_incl,:);
        ev_summary_interval = Dtb.get_ev_summary_intervals(C{:});
        
        %%
        util = Dtb.get_util_intervals(C{:});
        n_itv = size(util, 2);
        p_util = zeros(n_itv, 2, 2);
        for used1 = 0:1
            for used2 = 0:1
                for itv = n_itv:-1:1
                    incl = (util(:,itv,1) == used1) ...
                         & (util(:,itv,2) == used2);
                    p_util(itv, used1+1, used2+1) = ...
                        mean(incl);
                end
            end
        end
        Rev.p_util_sim = p_util;
        
        %%
        n_dim = 2;
        for dim = 1:n_dim
            subplot(1,n_dim, dim);
            imagesc(util(:,:,dim)); % DEBUG
        end
        
        %%
        C_noTnd = varargin2C({
            't_util_spec', {'all'}
            }, C);
        Dtb_noTnd = IxnKernel.RevCor.D2.SimDtbUtilD2(C_noTnd{:});
        Dtb_noTnd.simulate;
        Rev.Dtb_noTnd = Dtb_noTnd;
        
        %% make EP_noTnd
        EP_noTnd = IxnKernel.RevCor.D2.Ev2PchD2;
        EP_noTnd.Ev = Dtb_noTnd.Ev;
        EP_noTnd.fit_intervals(C{:});
        Rev.EP_noTnd = EP_noTnd;
        
        %%
        % p_ch(tr, itv, dim) = f_p_ch_given_util(ev(tr,itv), itv)
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
        % ch(tr, dim) : 0 or 1
        % ev_summary_interval(tr, itv, dim)
        % p_ch = f_p_ch_given_util(ev, n_passed+1)
        %
        % p_util_est_est(used1+1, used2+1, n_itv_passed+1)
        % p_util_est_ci(used1+1, used2+1, n_itv_passed+1, [lb, ub])
        % p_util_est_boot(used1+1, used2+1, n_itv_passed+1, boot)
        % p_util_est_by_n_used_lik(used1+1, used2+1, n_passed+1, n_used1+1, n_used2+1)
        % p_rest_by_n_used(n_passed+1, n_used1+1, n_used2+1)
        % p_util_jt(used1+1, used2+1, n_passed+1, n_used1+1, n_used2+1)
            
        n_boot = Rev.n_boot;
        n_tr = size(ch, 1);
        n_itv = size(ev_summary_interval, 2);
        
        p_util_est_boot = zeros(2, 2, n_itv, n_boot);
        
        t_st = tic;
        for i_boot = 1:n_boot % parfor
            if i_boot == 1
                % Use the original.
                ix = (1:n_tr)';
            else
                % Resample with replacement.
                ix = randi(n_tr, [n_tr, 1]);
            end
            ch1 = ch(ix,:);
            ev_summary_interval1 = ev_summary_interval(ix, :, :);
            
            [p_util_est1, p_util_est_by_n_used_lik1, ...
                p_rest_by_n_used1, p_util_jt1] = ...
                    ch2p_util_multi_t_bins_unit( ...
                        Rev, ch1, ev_summary_interval1, f_p_ch_given_util, ...
                        'to_cache', i_boot == 1);
                    
            p_util_est_by_n_used_lik_all{i_boot} = p_util_est_by_n_used_lik1;
            p_rest_by_n_used_all{i_boot} = p_rest_by_n_used1;
            p_util_jt_all{i_boot} = p_util_jt1;
            
            p_util_est_boot(:,:,:,i_boot) = p_util_est1;
            
            t_el = toc(t_st);
            fprintf('Bootstrap #1-#%d took %1.1f s\n', i_boot, t_el);
        end
        
        p_util_est_by_n_used_lik = p_util_est_by_n_used_lik_all{1};
        p_rest_by_n_used = p_rest_by_n_used_all{1};
        p_util_jt = p_util_jt_all{1};

        p_util_est_est = median(p_util_est_boot, 4);
        p_util_est_ci = prctile(p_util_est_boot, [2.5, 97.5], 4);
        
        Rev.p_util_est_est = p_util_est_est;
        Rev.p_util_est_ci = p_util_est_ci;
        Rev.p_util_est_boot = p_util_est_boot;
    end
    function [p_util_est, p_util_est_by_n_used_lik, ...
                p_rest_by_n_used, p_util_jt] = ...
            ch2p_util_multi_t_bins_unit( ...
                Rev, ch, ev_summary_interval, f_p_ch_given_util, varargin)
        % Maximum likelihood estimate of the momentary evidence use.
        %
        % [p_util_est, p_util_est_by_n_used] = ch2p_util_multi_t_bins_unit( ...
        %         Rev, ch, ev_summary_interval, f_p_ch_given_util, ...)
        %
        % ch(tr, dim) : 0 or 1
        % ev_summary_interval(tr, itv, dim)
        % p_ch = f_p_ch_given_util(ev, n_passed+1)
        %
        % p_util_est(used1+1, used2+1, n_itv_passed+1)
        % p_util_est_by_n_used_lik(used1+1, used2+1, n_passed+1, n_used1+1, n_used2+1)
        % p_rest_by_n_used(n_passed+1, n_used1+1, n_used2+1)
        % p_util_jt(used1+1, used2+1, n_passed+1, n_used1+1, n_used2+1)
        %
        % OPTIONS:
        % 'to_cache', true
        
        S = varargin2S(varargin, {
            'to_cache', true
            });
        
        n_itv = size(ev_summary_interval, 2);
        
        % p_util_est_by_n_used_lik ...
        %     (used1+1, used2+1, n_passed+1, n_used1+1, n_used2+1)
        p_util_est_by_n_used_lik = zeros(2, 2, n_itv, n_itv, n_itv);
        
        %% Estimate likelihood
        for n_passed = 0:(n_itv - 1)
            for n_used1 = 0:n_passed
                for n_used2 = 0:n_passed
                    % p_ch_given_util(tr, 1, used+1, dim)
                    p_ch_given_util = f_p_ch_given_util( ...
                        ev_summary_interval(:, n_passed + 1, :), ...
                        [n_used1, n_used2]);

                    % p_util_est_by_n_used_lik ...
                    %     (used1+1, used2+1, n_passed+1, n_used1+1, n_used2+1)
                    p_util_est_by_n_used_lik( ...
                        :, :, n_passed + 1, n_used1 + 1, n_used2 + 1) = ...
                        Rev.ch2p_util_unit( ...
                            ch, p_ch_given_util);
                end
            end
        end
        
        %% Prob remaining by n_passed using recurrence relation
        % p_rest_by_n_used(n_passed+1, n_used1+1, n_used2+1)
        p_rest_by_n_used = zeros(n_itv, n_itv, n_itv);
        p_rest_by_n_used(1,1,1) = 1;
        for n_passed = 0:(n_itv - 2)
            for n_used1 = 0:n_passed
                for n_used2 = 0:n_passed
                    p_rest1 = p_rest_by_n_used( ...
                        n_passed + 1, n_used1 + 1, n_used2 + 1);
                    
                    % p_util1(used1+1, used2+1)
                    p_util1 = p_util_est_by_n_used_lik( ...
                        :, :, n_passed + 1, n_used1 + 1, n_used2 + 1);

                    % Add depending on used1, 2.
                    for used1 = 0:1
                        for used2 = 0:1
                            ix_next = {
                                n_passed + 2, ...
                                n_used1 + 1 + used1, ...
                                n_used2 + 1 + used2
                                };
                            p_rest_by_n_used(ix_next{:}) = ...
                                p_rest_by_n_used(ix_next{:}) ...
                                + p_rest1 .* p_util1(used1 + 1, used2 + 1);
                        end
                    end
                end
            end
        end
        
        % p_util_jt(used1+1, used2+1, n_passed+1, n_used1+1, n_used2+1)
        p_util_jt = bsxfun(@times, ...
            p_util_est_by_n_used_lik, ...
            permute(p_rest_by_n_used, [4, 5, 1, 2, 3]));
        
        % p_util_est(used1+1, used2+1, n_passed+1)
        p_util_est = sums(p_util_jt, [4, 5]);
        
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
        %%
        import bml.plot.color_lines
        colors = {color_lines('b'), color_lines('r')
                  color_lines('g'), color_lines('k')};
        specs = {
            ':', ':'
            ':', '-'
            };
        
        %%
        clf;
        hs = ghandles(2,2);
        for used1 = 0:1
            for used2 = 0:1
                p_util = squeeze(Rev.p_util_est(used1+1, used2+1, :));
                n = numel(p_util);
                hs(used1+1,used2+1) = ...
                    plot(1:n, p_util, ...
                        'Color', colors{used1+1, used2+1}, ...
                        'LineStyle', specs{used1+1, used2+1}, ...
                        'LineWidth', 2);
                hold on;
            end
        end
        hold off;
        
        legend(hs(:), {'0,0', '1,0', '0,1', '1,1'});
        bml.plot.beautify;
              
        %%
        file = Rev.get_file({'plt', 'p_util_all'});
        savefigs(file);
    end
    function plot_p_util_boot_all(Rev, varargin)
        S = varargin2S(varargin, {
            'mark_indep', true % if true, mark the independent case
            'mark_marginal', true % if true, mark the marginal frequency in 'alone'
            'style', 'violin' % 'violin'|'errorbar'
            'style_short', 'vln' % 'vln'|'err'
            });

        style = S.style;
        style_short = S.style_short;
        
        files = cell(2,2);
        for use1 = 0:1
            for use2 = 0:1
                clf;
                
                to_print_pval = false;
                if S.mark_indep && use1 == 1 && use2 == 1
                    to_print_pval = true;
                    
                    p_sim = sum(Rev.p_util_sim(:, 2, :), 3) ...
                         .* sum(Rev.p_util_sim(:, :, 2), 2);
                     
                    n_itv = numel(p_sim);
                    itv = 1:n_itv; % 0:(n_itv-1);
                    
                    x_st = [0.5; itv(:) + 0.5];
                    y_st = [p_sim(:); p_sim(end)];
                    
                    hold on;
                    stairs(x_st + 0.03, y_st + 0.005, 'r--', 'LineWidth', 2);
                    hold on;
                    
%                     p = squeeze(bsxfun(@times, ....
%                             sum(Rev.p_util_est_est(:,2,:), 1), ...
%                             sum(Rev.p_util_est_est(2,:,:), 2)));
%                     p_all = bsxfun(@times, ...
%                             sum(Rev.p_util_est_boot(:, 2, :, :), 1), ...
%                             sum(Rev.p_util_est_boot(2, :, :, :), 2));
%                     p_all = squeeze(permute(p_all, [4, 3, 1, 2]));
%                     
%                     ci = [
%                         vVec(prctile(p_all, 2.5, 1)), ...
%                         vVec(prctile(p_all, 97.5, 1))
%                         ];
%                     
%                     hold on;
%                     Rev.plot_p_util_boot( ...
%                         'used', [use1, use2], ...
%                         'style', style, ...
%                         'p', p, ...
%                         'p_all', p_all, ...
%                         'e', ci, ...
%                         'p_sim', [], ...
%                         'color', 'r', ...
%                         'x_offset', 0.2);
                elseif S.mark_marginal && use1 + use2 == 1
                    to_print_pval = true;
                    
                    if use1 == 1
                        p_sim = sum(Rev.p_util_sim(:, 2, :), 3);
                    else
                        p_sim = sum(Rev.p_util_sim(:, :, 2), 2);
                    end
                     
                    n_itv = numel(p_sim);
                    itv = 1:n_itv; % 0:(n_itv-1);
                    
                    x_st = [0.5; itv(:) + 0.5];
                    y_st = [p_sim(:); p_sim(end)];
                    
                    hold on;
                    stairs(x_st + 0.03, y_st + 0.005, '--', ...
                        'Color', bml.plot.color_lines('b'), ...
                        'LineWidth', 2);
                    hold on;
                    
%                     if use1 == 1 % && use2 == 0
%                         p = squeeze( ...
%                             sum(Rev.p_util_est_est(2,:,:), 2));
%                         p_all = ...
%                             sum(Rev.p_util_est_boot(2,:,:,:), 2);
%                     else % use1 == 0 && use2 == 1
%                         p = squeeze( ...
%                             sum(Rev.p_util_est_est(:,2,:), 1));
%                         p_all = ...
%                             sum(Rev.p_util_est_boot(:,2,:,:), 1);
%                     end
%                     
%                     p_all = squeeze(permute(p_all, [4, 3, 1, 2]));
%                     ci = [
%                         vVec(prctile(p_all, 2.5, 1)), ...
%                         vVec(prctile(p_all, 97.5, 1))
%                         ];
%                     
%                     hold on;
%                     Rev.plot_p_util_boot( ...
%                         'used', [use1, use2], ...
%                         'style', style, ...
%                         'p', p, ...
%                         'p_all', p_all, ...
%                         'e', ci, ...
%                         'p_sim', [], ...
%                         'color', bml.plot.color_lines('b'), ...
%                         'x_offset', 0.2);
                end
                
                [~, ~, ~, p_all, p_sim0] = Rev.plot_p_util_boot( ...
                    'used', [use1, use2], ...
                    'style', style);                                     
                                                
                file = Rev.get_file({
                    'plt', style_short
                    'use', [use1, use2]
                    });
                
                if to_print_pval
                    mkdir2(fileparts(file));
                    file_txt = [file, '.txt'];
                    if exist(file_txt, 'file')
                        delete(file_txt);
                    end
                    diary(file_txt);

                    disp('use1, 2:');
                    disp([use1, use2]);
                    
                    % comparison between difference from the 
                    % true vs marginal
                    pval = mean( ...
                        abs(bsxfun(@minus, p_all', p_sim0)) ...
                        >= abs(bsxfun(@minus, p_all', p_sim)), 2);
                    disp('pval (dif_from_true >= dif_from_marginal):');
                    disp(pval);
                    
                    % difference from the marginal case
%                     pval = min( ...
%                         mean(bsxfun(@lt, p_all', p_sim), 2), ...
%                         mean(bsxfun(@gt, p_all', p_sim), 2)) * 2;
%                     disp('pval:');
%                     disp(pval);
                    
                    diary off
                end
                
                savefigs(file, 'size', [400, 200]);

                files{use1 + 1, use2 + 1} = [file, '.fig'];
            end
        end
        
        %% Imgather
        clf;
        ax = subplotRCs(2, 2);
        for use1 = 0:1
            for use2 = 0:1
                ax1 = ax(use1+1, use2+1);
                file1 = files{use1+1, use2+1};
                
                [ax1, h] = openfig_to_axes(file1, ax1);
                ax(use1+1, use2+1) = ax1;
            end
        end
        
        %% Beautify
        for use1 = 0:1
            for use2 = 0:1
                ax1 = ax(use1+1, use2+1);
                
                if ~(use1 == 1 && use2 == 0)
                    xlabel(ax1, '');
                end
                
                h_legend = legend(ax1);
%                 if ~(use1 == 1 && use2 == 0)
                    delete(h_legend);
%                 else
%                     set(h_legend, 'Location', 'NorthEast');
%                 end
                
                if use1 == 0
                    set(ax1, 'XTickLabel', '');
                end
                if use2 == 1
                    set(ax1, 'YTickLabel', '');
                end
            end
        end
        
        %% Previous format
%         bml.plot.position_subplots(ax, ...
%             'margin_left', 0.1, ...
%             'margin_bottom', 0.15, ...
%             'margin_top', 0.05);
%         file = Rev.get_file({
%             'plt', ['igth_' style_short]
%             });
%         savefigs(file, 'size', [400, 300], ...
%             'ext', {'.png', '.tif', '.fig'});
        
        %% New format with space for titles rather than ylabel
        bml.plot.position_subplots(ax, ...
            'margin_left', 0.1, ...
            'margin_bottom', 0.15, ...
            'margin_top', 0.05, ...
            'btw_row', 0.15, ...
            'btw_col', 0.03);
        
        %%
        for ii = 1:numel(ax)
            ylabel(ax(ii), '');
        end
        
        %%
        file = Rev.get_file({
            'plt', ['igth_' style_short]
            });
        savefigs(file, 'size', [267, 253] * 1.25, ... % [400, 300], ...
            'ext', {'.png', '.tif', '.fig'});
    end
    function [p, e, p_all, p_sim] = get_p_util(Rev, used)
        % [p, e, p_all, p_sim] = get_p_util(Rev, used)
        
        p = squeeze(Rev.p_util_est_est(used(1)+1, used(2)+1, :));
        ci = squeeze(Rev.p_util_est_ci(used(1)+1, used(2)+1, :, :));
        e = bsxfun(@minus, ci, p);
        
        p_all = permute( ...
            Rev.p_util_est_boot(used(1)+1, used(2)+1, :, :), ...
            [4, 3, 1, 2]);
        
        if ~isempty(Rev.p_util_sim)
            p_sim = Rev.p_util_sim(:, used(1)+1, used(2)+1);
        end
    end
    function [h, he, h_orig, p_all, p_sim] = plot_p_util_boot(Rev, varargin)
        %%
        S = varargin2S(varargin, {
            'used', [1, 1]
            'style', 'errorbar' % 'errorbar', 'violin'
            'color', 'k'
            'alpha', 0.3
            'x_offset', 0
            'p', []
            'e', []
            'p_all', []
            'p_sim', []
            });
        
        if isempty(S.p)
            [p, e, p_all, p_sim] = Rev.get_p_util(S.used);
        else
            p = S.p;
            e = S.e;
            p_all = S.p_all;
            p_sim = S.p_sim;
        end

        n_itv = numel(p);
        itv = 1:n_itv; % 0:(n_itv-1);
        
        hs_for_legend = [];
        legends = {};
        
        switch S.style
            case 'errorbar'
                if ~isempty(p_sim)
                    hold on;
                    x_st = [0.5; itv(:) + 0.5];
                    y_st = [p_sim(:); p_sim(end)];
                    h_orig = stairs(x_st, y_st, 'k-', 'LineWidth', 2);
%                     h_orig = plot(itv + 1, p_sim, 'k-', 'LineWidth', 2);
                    
                    hs_for_legend = [hs_for_legend(:); h_orig];
                    legends = {'Estimate'; 'True value'};
                end

                [h, he] = errorbar_wo_tick(itv + S.x_offset, p, e(:,1), e(:,2), {
                        'Marker', 'o'
                        'MarkerEdgeColor', 'w'
                        'MarkerFaceColor', S.color
                        });
                ylim([-0.01, 1.01]);
                xlim([0.5, n_itv + 0.5]);
%                 xlim([-0.5, n_itv - 0.5]);
                hold off;

            case 'violin'
                % p_all(boot, itv)
                hs = distributionPlot(p_all, ...
                    'color', S.color);
                he = hs{1};
                h = hs{2};
                set(h, 'Color', S.color);
                set(h(2), 'Marker', 'x');
                set(he, 'FaceColor', S.color, ...
                        'FaceAlpha', S.alpha);
                
                hs_for_legend = h;
                
                set(gca, ...
                    'XTickLabel', (1:n_itv) - 1);
                xlim([0.5, n_itv + 0.5]);
                ylim([-0.01, 1.01]);
                
                if ~isempty(p_sim)
                    hold on;
                    x_st = [0.5; itv(:) + 0.5];
                    y_st = [p_sim(:); p_sim(end)];
                    h_orig = stairs(x_st, y_st, 'k-', 'LineWidth', 2);
%                     h_orig = plot(itv + 1, p_sim, 'k-', 'LineWidth', 2);
                    hold off;
                    
                    hs_for_legend = [hs_for_legend(:); h_orig];
                    legends = [legends(:); {'True value'}];
                end
        end
        
        if ~isempty(p_sim) % hs_for_legend)
            legend(hs_for_legend, legends, ...
                'Location', 'NorthEastOutside');
        end
        
        y_labels = {
            'none',  'C only'
            'M only','both'
            };
        y_label = y_labels{S.used(1)+1, S.used(2)+1};
        
        ylabel(sprintf('P_{%s}', y_label));
        xlabel('Interval (\tau)');
        bml.plot.beautify;
        
        set(gca, 'YTick', 0:0.25:1, 'YTickLabel', {'0', '', '', '', '1'});
    end    
end
%% One time bin
methods
    function p_util_est = ch2p_util_unit(Rev, ch, p_ch_given_util)
        % p_ch_given_util(tr, itv, dim) = P(ch(tr)=1 | util=1, dim)
        %
        % p_util_est(used1+1,used2+1)
        
        % p_ch_given_util must be (tr, 1, dim)
        assert(size(p_ch_given_util, 3) == 2);
        
        %% Cost function
        [ch, p_ch_given_util] = bml.matrix.rows_w_no_nan(ch, p_ch_given_util);
        
        f = @(p_util) -Rev.get_log_p_p_util_given_ch( ...
            p_util, ch, p_ch_given_util);
%         disp(f(0.5)); % Test run
        
        %% Estimate p_util given p_ch_util and ch
        epsilon = 1e-6;
        x0 = [0.25, 0.25, 0.25];
        lb = epsilon + zeros(1, 3);
        ub = ones(1, 3) - epsilon * 3;
        
        A = [1 1 1];
        b = 1;
        
        %% Test run
%         cost = f(x0);
%         disp(cost); % DEBUG
        
        %% Run
        opt = optimoptions('fmincon', 'Display', 'notify');
        [p_util_est3, fval, exitflag, output, lambda, grad, hessian] = ...
            fmincon(f, ...
            x0, A, b, [], [], ...
            lb, ub, [], opt);
        
%         % DEBUG
%         x = 0.1:0.01:0.9;
%         plot(x, f(x));
%         disp(p_util_est3); % DEBUG
        
        p_util_est(1,1) = 1 - sum(p_util_est3);
        p_util_est(2,2) = p_util_est3(1);
        p_util_est(2,1) = p_util_est3(2);
        p_util_est(1,2) = p_util_est3(3);
        
%         disp(p_util_est); % DEBUG
    end
    function log_p_p_util = get_log_p_p_util_given_ch(~, ...
            p_util0, ch, p_ch_given_util)
        % log_p_p_util = get_log_p_p_util_given_ch(Rev, ...
        %             p_util0, ch, p_ch_given_util)        
        %
        % p_util0([used_both, used_1_only, used_2_only]
        % ch(tr,dim) : 0 or 1
        % p_ch_given_util(tr, 1, dim) ...
        %     = P(ch(tr,dim)=+1 | util(tr,dim)=1)
        %
        % log_p_p_util : scalar
        
        epsilon = 1e-6;
        p_ch_given_util = min(max(p_ch_given_util, epsilon), 1 - epsilon*3);
        p_util0 = min(max(p_util0, epsilon), 1 - epsilon*3);
        p_util(2,2) = p_util0(1); % Used both
        p_util(2,1) = p_util0(2); % Used dim 1 only
        p_util(1,2) = p_util0(3); % Used dim 2 only
        p_util(1,1) = 1 - sum(p_util0); % Used neither
        
        % p_ch_given_util(tr, ch, dim) ...
        %     = P(ch(tr,dim)=ch | util(tr,dim)=1)
        p_ch_given_util = cat(2, ...
            1 - p_ch_given_util, ...
            p_ch_given_util);
        
        % p_ch_given_util(tr, ch, dim, use+1) ...
        %     = P(ch(tr,dim)=ch | util(tr,dim)=use)
        p_ch_given_util = cat(4, ...
            zeros(size(p_ch_given_util)) + 0.5, ...
            p_ch_given_util);
        
        n_tr = size(ch, 1);
        
        % p_ch_obs_given_util(tr, 1, ch1, ch2)
        p_ch_obs_given_util0 = zeros(n_tr, 1, 2, 2);        
        
        for ch1 = 1:2
            for ch2 = 1:2
                for use1 = 1:2
                    for use2 = 1:2
                        p_ch_obs_given_util0(:,1,ch1,ch2) = ...
                            p_ch_obs_given_util0(:,1,ch1,ch2) + ...
                            p_util(use1, use2) ...
                            .* p_ch_given_util(:,ch1,1,use1) ...
                            .* p_ch_given_util(:,ch2,2,use2);
                    end
                end
            end
        end
        
        p_ch_obs_given_util = zeros(n_tr, 1);
        for ch1 = 0:1
            for ch2 = 0:1
                incl = (ch(:,1) == ch1) & (ch(:,2) == ch2);
                p_ch_obs_given_util(incl) = ...
                    p_ch_obs_given_util0(incl,1,ch1+1,ch2+1);
            end
        end
        
        log_p_p_util = sum(log(p_ch_obs_given_util));
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
        for style = {
                'errorbar', 'err'
                'violin',   'vln'
                }'
            Rev.plot_p_util_boot_all( ...
                'style', style{1}, ...
                'style_short', style{2});
        end
    end
    function plot_sim_unit(Rev, style, style_short)
        % plot_sim_unit(Rev, style, style_short)
        
        if nargin < 2
            style = 'violin';
            style_short = 'vln';
        end
        
        files = cell(2,2);
        for use1 = 0:1
            for use2 = 0:1
                clf;
                Rev.plot_p_util_boot( ...
                    'used', [use1, use2], ...
                    'style', style);                     
                file = Rev.get_file({
                    'plt', style_short
                    'use', [use1, use2]
                    });
                savefigs(file, 'size', [400, 200]);

                files{use1 + 1, use2 + 1} = [file, '.fig'];
            end
        end
        
        %% Imgather
        clf;
        ax = subplotRCs(2, 2);
        for use1 = 0:1
            for use2 = 0:1
                ax1 = ax(use1+1, use2+1);
                file1 = files{use1+1, use2+1};
                
                [ax1, h] = openfig_to_axes(file1, ax1);
                ax(use1+1, use2+1) = ax1;
            end
        end
        
        %% Beautify
        for use1 = 0:1
            for use2 = 0:1
                ax1 = ax(use1+1, use2+1);
                
                if ~(use1 == 1 && use2 == 0)
                    xlabel(ax1, '');
                end
                
                h_legend = legend(ax1);
%                 if ~(use1 == 1 && use2 == 0)
                    delete(h_legend);
%                 else
%                     set(h_legend, 'Location', 'NorthEast');
%                 end
                
                if use1 == 0
                    set(ax1, 'XTickLabel', '');
                end
                if use2 == 1
                    set(ax1, 'YTickLabel', '');
                end
            end
        end
        
        %% Previous format
%         bml.plot.position_subplots(ax, ...
%             'margin_left', 0.1, ...
%             'margin_bottom', 0.15, ...
%             'margin_top', 0.05);
%         file = Rev.get_file({
%             'plt', ['igth_' style_short]
%             });
%         savefigs(file, 'size', [400, 300], ...
%             'ext', {'.png', '.tif', '.fig'});
        
        %% New format with space for titles rather than ylabel
        bml.plot.position_subplots(ax, ...
            'margin_left', 0.1, ...
            'margin_bottom', 0.15, ...
            'margin_top', 0.05, ...
            'btw_row', 0.15, ...
            'btw_col', 0.03);
        
        %%
        for ii = 1:numel(ax)
            ylabel(ax(ii), '');
        end
        
        %%
        file = Rev.get_file({
            'plt', ['igth_' style_short]
            });
        savefigs(file, 'size', [267, 253] * 1.25, ... % [400, 300], ...
            'ext', {'.png', '.tif', '.fig'});
        
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
    function batch_fit(Rev, varargin)
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            });
        [Ss, n] = factorizeS(S_batch);
        
        for ii = 1:n
            C = S2C(Ss(ii));
            Rev.main_fit(C{:});
        end
    end
    function main_fit(Rev, varargin)
        % main_fit(Rev, varargin)
        %
        % Options: 
        % Any options as in one of the following:
        % - Rev.get_S_Dtb_EP()
        % - Rev.import_pdf
        %   - dif_rel_incl for EvTimeD1Pdf.import_pdf
        %   - dif_incl for EvTimeD2.import_data
        
        files = Rev.batch_export_pdfs(varargin{:});
        n = size(files, 1);
        for ii = 1:n
            Rev.fit_aft_import(files(ii,:), varargin{:});
        end
    end
    function files = batch_export_pdfs(Rev, varargin)
        % files = batch_export_pdfs(Rev, varargin)
        %
        % files{subj, [1D_dim1, 1D_dim2, 2D]} = pdf_file
        
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            });
        [Ss, n] = factorizeS(S_batch);
        
        assert(~isfield(S_batch, 'dim_rel_W'), ...
            'dim_rel_W must be left unspecified!');
        n_dim = Data.Consts.n_dim;
        
        files = cell(n, n_dim + 1);
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            
            % 1D pred exports
            W0 = Fit.D1.BoundedEn.Main;
            files_1D = W0.batch_export_pdf(C{:}, ...
                'dim_rel_W', 1:n_dim);
            files(ii, 1:n_dim) = files_1D(:)';
            
            % 2D data exports
            W0 = Fit.D2.Common.DataFilterEn;
            files_2D = W0.batch_export_data(C{:});
            files(ii, n_dim + 1) = files_2D;
        end
    end
    function batch_fit_aft_import(Rev, varargin)
        S_batch = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT
            });
        [Ss, n] = factorizeS(S_batch);
        
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            
            Rev.fit_aft_import([], C{:});
        end
    end
    function fit_aft_import(Rev, files, varargin)
        % fit_aft_import(Rev, files, varargin)
        %
        % files{[1D_dim1, 1D_dim2, 2D]} = pdf_file
        % : as exported by Rev.batch_export_pdfs
        %
        % Options: 
        % Any options as in one of the following:
        % - Rev.get_S_Dtb_EP()
        % - Rev.import_pdf
        %   - dif_rel_incl for EvTimeD1Pdf.import_pdf
        %   - dif_incl for EvTimeD2.import_data
        
        S = varargin2S(varargin, {
            'subj', Data.Consts.subjs_RT{1}
            });
        
        %% Default files
        if ~exist('files', 'var') || isempty(files)
%             files = {
%                 '../Data_2D/Fit.D1.BoundedEn.Main/sbj=DX+prd=RT+tsk=H+dtk=1+dmr=1+trm=201+eor=t+bnd=C+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=1+fbst=1+exp=pred.mat'
%                 '../Data_2D/Fit.D1.BoundedEn.Main/sbj=DX+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+bnd=C+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=1+fbst=1+exp=pred.mat'
%                 '../Data_2D/Fit.D2.Common.DataFilterEn/t0=st+trc_s=-Inf+trc_e=0+smth=0^05+sbj=DX+prd=RT+nd_tsk=2+tsk=A+dim_r=1+dif_i=[1,2,3]+acc_i=[0,1]+dif_r=[1,2,3]+dfr=[1,2,3]+dfi=[1,2,3]+aci=[0,1]+trm=201+eor=t+exp=data.mat'
%                 };
            files = {
                '../Data_2D/Fit.D1.BndEn2Bnd.Main/sbj=DX+prd=RT+tsk=A+dtk=2+dmr=1+trm=201+eor=t+dft=C+bnd=A2+ssq=C+tnd=i+ntnd=2+msf=1+fsqs=1+frst=10+fbst=0+exp=pred.mat'
                '../Data_2D/Fit.D1.BndEn2Bnd.Main/sbj=DX+prd=RT+tsk=A+dtk=2+dmr=2+trm=201+eor=t+dft=C+bnd=A2+ssq=C+tnd=i+ntnd=2+msf=1+fsqs=1+frst=10+fbst=0+exp=pred.mat'
                '../Data_2D/Fit.D2.Common.DataFilterEn/t0=st+trc_s=-Inf+trc_e=0+smth=0^05+sbj=MA+prd=RT+nd_tsk=2+tsk=A+dim_r=1+dif_i=all+acc_i=[0,1]+dif_r=all+aci=[0,1]+trm=201+eor=t+exp=data.mat'
%                 '../Data_2D/Fit.D2.Common.DataFilterEn/t0=st+trc_s=-Inf+trc_e=0+smth=0^05+sbj=DX+prd=RT+nd_tsk=2+tsk=A+dim_r=1+dif_i=[1,2,3]+acc_i=[0,1]+dif_r=[1,2,3]+dfr=[1,2,3]+dfi=[1,2,3]+aci=[0,1]+trm=201+eor=t+exp=data.mat'
                };
        end
        files = strrep_cell(files, {
            'sbj=DX', sprintf('sbj=%s', S.subj)
            });
        
        %% Import
        [ch, ev_summary_interval, f_p_ch_given_util] = ...
            Rev.import_pdf(files, varargin{:});
        
        %% Bootstrap
        [p_util_est_est, p_util_est_ci, p_util_est_boot, ...
            p_util_est_by_n_used_lik, p_rest_by_n_used, p_util_jt] = ...
                Rev.ch2p_util_multi_t_bins_boot( ...
                    ch, ev_summary_interval, f_p_ch_given_util);
        
        %% Plot
        Rev.plot_sim;
        
        %% Save
        Rev.save_mat;
    end
    function [ch, ev_summary_interval, f_p_ch_given_util] = ...
            import_pdf(Rev, files, varargin)
        % Import RT_data and Td_pred to Dtb and EP.
        % Specifically, RT_data_pdf_tr goes to Dtb and EP,
        % and Td_pred_pdf_tr goes to Dtb_noTnd and EP_noTnd.
        %
        % import_pdf(Rev, files)
        %
        % files{[1D_dim1, 1D_dim2, 2D]}
        % 1D files provide no_Tnd fits.

        %%
        Rev.Dtb = IxnKernel.RevCor.D2.SimDtbUtilD2;
        Rev.Dtb.Ev = IxnKernel.EvTime.EvTimeD2;
        Rev.Dtb.Ev.import_data(files{3}, varargin{:}); % 2D data
        
        %%
        Rev.Dtb_noTnd = IxnKernel.RevCor.D2.SimDtbUtilD2;
        Rev.Dtb_noTnd.Ev = IxnKernel.EvTime.EvTimeD2Pdf;
        for dim = 1:2
            Rev.Dtb_noTnd.Ev.Evs{dim} = IxnKernel.EvTime.EvTimeD1Pdf;
            Rev.Dtb_noTnd.Ev.Evs{dim}.import_pdf(files{dim}, ...
                'src', 'Td_pred', varargin{:});
        end
        
        %%
        Rev.Ev = Rev.Dtb.Ev;
        Rev.ch = Rev.Dtb.Ev.ch;
        Rev.rt_fr = Rev.Dtb.Ev.rt_fr;
        Rev.ev_summary_interval = Rev.Dtb.get_ev_summary_intervals;
        Rev.util = []; % Unknown
        
        Rev.EP = IxnKernel.RevCor.D2.Ev2PchD2;
        Rev.EP.Ev = Rev.Dtb.Ev;
        
        Rev.EP_noTnd = IxnKernel.RevCor.D2.Ev2PchD2;
        Rev.EP_noTnd.Ev = Rev.Dtb_noTnd.Ev;
        
        C = S2C(Rev.get_S_Dtb_EP(varargin{:}));
        Rev.EP_noTnd.fit_intervals(C{:});
        
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