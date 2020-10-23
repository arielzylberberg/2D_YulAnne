classdef Main ...
        < Fit.Common.Main ...
        & Fit.D1.Common.CommonWorkspace
    % Fit.D1.Bounded.Main
    %
    % 2015 YK wrote the initial version.
%% Internal
properties (SetAccess = protected)
    % Data % Defined in Ws.
    Dtb
    Tnd
    Miss
end
%% Main
methods
    function W = Main(varargin)
        % Set defaults
        W.set_Data;
        W.set_Dtb;
        W.set_Tnd;
        W.set_Miss;
        
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        W.init@Fit.D1.Common.CommonWorkspace(varargin{:});
        W.constrain_tnd_from_data;
    end
    function main(W)
        min_mean_rt = min(accumarray(W.Data.dCond, W.Data.rt, [], @mean));
        min_std_rt = min(accumarray(W.Data.dCond, W.Data.rt, [], @std));
        
        W.Tnd.set_mus(min_mean_rt * 0.5);
        W.Tnd.set_sds(min_std_rt * 0.5);
        W.Tnd.th0 = W.Tnd.th;
        
        W.fit;
        W.save_mat;
        W.plot_and_save_all;
    end
    function pred(W)
        W.Dtb.pred;
        W.Tnd.pred;
        W.Miss.pred;
    end
    function [c, c_sep] = calc_cost(W)
        pred = W.Data.get_RT_pred_pdf;
        data = W.Data.get_RT_data_pdf;
        
        siz0 = size(data);
        siz = [siz0(1) * siz0(3), siz0(2)];
        
        f_reshape = @(v) reshape(permute(v, [1, 3, 2]), siz);
        pred = f_reshape(pred);
        data = f_reshape(data);
        
        [c, c_sep] = nll_bin( ...
            pred, ...
            data, 'normalize', true);
    end
    function [c, c_sep] = calc_cost_per_trial(W)
        [~, ~, d_cond] = unique(W.Data.cond);
        
        RT_pred_pdf = W.Data.get_RT_pred_pdf;
        RT_data_pdf = W.Data.get_RT_data_pdf;
        
        RT_pred_pdf_tr = RT_pred_pdf(:, d_cond, :);
        RT_data_pdf_tr = RT_data_pdf(:, d_cond, :);
        
        [c, c_sep] = nll_bin( ...
            RT_pred_pdf_tr, ...
            RT_data_pdf_tr);
    end
end
%% Model setting
methods
    function v = obj2kind(~, objs, obj_kind)
        f = @(c) strrep(bml.pkg.pkg2class(c), obj_kind, '');
        v = {f(objs{1}), f(objs{2})};
        if strcmp(v{1}, v{2})
            v = v{1};
        end
    end
    
    function v = get_drift_kind(W)
        v = W.Dtb.Drift.kind;
    end
    function set_drift_kind(W, v)
        W.Dtb.set_Drift(v);
    end
    
    function v = get_bound_kind(W)
        v = W.Dtb.Bound.kind;
    end
    function set_bound_kind(W, v)
        W.Dtb.set_Bound(v);
    end
    
    function v = get_sigmaSq_kind(W)
        v = W.Dtb.SigmaSq.kind;
    end
    function set_sigmaSq_kind(W, v)
        W.Dtb.set_SigmaSq(v);
    end
    
    function v = get_tnd_distrib(W)
        v = W.Tnd.distrib;
    end
    function set_tnd_distrib(W, v)
        v0 = W.Tnd.distrib;
        if ~strcmp(v0, v)
            W.Tnd.distrib = v;
            W.Tnd.init_params0;
        end
    end
    
    function v = get_n_tnd(W)
        v = W.Tnd.n_Tnd;
    end
    function v = set_n_tnd(W, v)
        v0 = W.Tnd.n_Tnd;
        if v0 ~= v
            W.Tnd.n_Tnd = v;
            W.Tnd.init_params0;
        end
    end
end
%% Set/Get object properties
methods
    function set_Data(W, obj_or_name)
        if nargin < 2, obj_or_name = Fit.D1.Common.DataChRtPdf; end
        obj_or_name = W.enforce_class( ...
            'Fit.D1.Common.DataChRtPdf', obj_or_name, {});
        W.set_Data@Fit.Common.CommonWorkspace(obj_or_name);
    end
    function set_Dtb(W, obj_or_name)
        if nargin < 2, obj_or_name = Fit.D1.Bounded.Dtb; end
        W.Dtb = W.enforce_class('Fit.D1.Bounded.Dtb', obj_or_name);
        W.set_sub_from_props({'Dtb'});
    end
    function set_Tnd(W, obj_or_name)
        if nargin < 2, obj_or_name = Fit.D1.Bounded.Tnd; end
        W.Tnd = W.enforce_class('Fit.D1.Bounded.Tnd', obj_or_name);
        W.set_sub_from_props({'Tnd'});
    end
    function set_Miss(W, obj_or_name)
        if nargin < 2, obj_or_name = Fit.Common.Miss; end
        W.Miss = W.enforce_class('Fit.Common.Miss', obj_or_name);
        W.set_sub_from_props({'Miss'});
    end
end
%% Parameter estimates
methods
    function [th1, lb1, ub1] = get_res_param(W, name)
        if ~W.Fl.is_valid_res
            th1 = nan;
            lb1 = nan;
            ub1 = nan;
            return;
        end
        
        th = W.Fl.res.th;
        se = W.Fl.res.se;
        
        lb1 = [];
        ub1 = [];
        
        switch name
            case {'kb', 'bkratio'}
                name1 = ['Dtb__' name];
                if isfield(th, name1)
                    th1 = th.(name1);
                    se1 = se.(name1);
                else
                    name1 = ['Dtb__log10_' name];
                    if isfield(th, name1)
                        th1 = th.(name1);
                        se1 = se.(name1);
                    end
                    
                    th1 = 10.^th1;
                    lb1 = 10.^(th1 - se1);
                    ub1 = 10.^(th1 + se1);
                end
            
            case 'k'
                th1 = th.Dtb__Drift__k;
                se1 = se.Dtb__Drift__k;
                
            case 'b'
                th1 = bml.math.means(abs(W.Dtb.Bound.get_bound_t_ch));
                [~, samp] = W.Dtb.Bound.get_se_bound_t_ch;
                th_samp = bml.math.means(abs(samp), [1, 2], true);
                se1 = sem(th_samp);
                
            case 'b_experienced'
                n_samp = 5e2;
                W.Fl.randsample(n_samp);
                
                b_t_ch = abs(W.Dtb.Bound.get_bound_t_ch);
                sub = [W.Data.get_RT_ix, W.Data.ds.ch(:, W.dim_rel_W) + 1];
                n_ch = 2;
                ix = sub2ind([W.nt, n_ch], sub(:,1), sub(:,2));
                b0 = b_t_ch(ix);
                
                th1 = median(b0);
                lb1 = prctile(b0, normcdf(-1)*100);
                ub1 = prctile(b0, normcdf(1)*100);
                
            case 'bias_cond'
                th1 = th.Dtb__Drift__bias;
                se1 = se.Dtb__Drift__bias;
                
            case 'bias_bound'
                th1 = th.Dtb__Bound__bias;
                se1 = se.Dtb__Bound__bias;
                
            case 'tnd_mu'
                th1 = (th.Tnd__mu_1 + th.Tnd__mu_2) / 2;
                
                se1 = sqrt(se.Tnd__mu_1.^2/4 + se.Tnd__mu_2.^2/4 ...
                    + W.Fl.get_cov([], 'Tnd__mu_1', 'Tnd__mu_2') / 2);
                
            case 'tnd_sd'
                n_samp = 5e2;
                W.Fl.randsample(n_samp);
                sd = zeros(n_samp, 1);
                th_orig = W.Tnd.th;
                for ii = n_samp:-1:1
                    W.Tnd.th_vec = W.Tnd.th_samp(ii, :);
                    sd(ii) = mean(W.Tnd.get_sds);
                end
                W.Tnd.th = th_orig;
                
%                 th1 = mean(W.Tnd.get_sds);
                th1 = median(sd);
                lb1 = prctile(sd, normcdf(-1)*100);
                ub1 = prctile(sd, normcdf(1)*100);
                
            case 'tnd_disper'
                th1 = (th.Tnd__disper_1 + th.Tnd__disper_2) / 2;
                
                se1 = sqrt(se.Tnd__disper_1.^2/4 + se.Tnd__disper_2.^2/4 ...
                    + W.Fl.get_cov([], 'Tnd__disper_1', 'Tnd__disper_2') / 2);
                
            case 'miss'
                if isfield(th, 'Miss__miss')
                    th1 = th.Miss__miss;
                    se1 = se.Miss__miss;
                else
                    th1 = th.Miss__logit_miss;
                    se1 = se.Miss__logit_miss;
                    
                    lb1 = invLogit(th1 - se1);
                    ub1 = invLogit(th1 + se1);
                    th1 = invLogit(th1);
                end
                
            case 'ssq_st'
                th1 = th.Dtb__log10_sigmaSq_st;
                se1 = se.Dtb__log10_sigmaSq_st;
                
                lb1 = 10.^(th1 - se1);
                ub1 = 10.^(th1 + se1);
                th1 = 10.^th1;
                
            case 'log10_ssq_max_cond'
                th1 = th.Dtb__SigmaSq__log10_sigmaSq_max_cond;
                se1 = se.Dtb__SigmaSq__log10_sigmaSq_max_cond;
                
            case 'ssq_max_cond'
                th1 = th.Dtb__SigmaSq__log10_sigmaSq_max_cond;
                se1 = se.Dtb__SigmaSq__log10_sigmaSq_max_cond;
                
                lb1 = th1 - se1;
                ub1 = th1 + se1;
                
                th1 = 10.^th1;
                lb1 = 10.^lb1;
                ub1 = 10.^ub1;
                
            otherwise
                th1 = th.(name);
                se1 = se.(name);
        end
        if isempty(lb1)
            lb1 = th1 - se1; 
        end
        if isempty(ub1)
            ub1 = th1 + se1;
        end
    end
end
%% Fit unit (deprecated) 
methods
    function S_fit_unit = get_S_fit_unit_default(~)
        S_fit_unit = varargin2S({
            'subj', Data.Consts.subjs_RT{1}
            'parad', 'RT'
            'task', 'H'
            ... % Dtb
            'Drift', 'Const' % 'DriftIndiv' % 'DriftPower' % 
            'Bound', 'Const'
            'SigmaSq', 'Const'
            'W_th0', []
            ...
            'adCond_incl', ':' % integer stimulus strength. 1, 2, ...
            'fix_params', false
            ...
            'fit_args', {}
            });
    end
    function add_plotfun(W, Fl)
        if nargin < 2
            Fl = W.get_Fl;
        end
        
        Fl.remove_plotfun_all;
        W.add_plotfun@Fit.D1.Common.CommonWorkspace(Fl)
        Fl.add_plotfun({
            @(Fl) @(x,v,s) ...
                Fit.D2.Common.Plot.PlotFuns.history_sum_p_pred(Fl, x, v, s)
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_ch)
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_rt)
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_rt('y_fun', 'var'))
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_rt('y_fun', 'std'))
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_bound)
            @(Fl) @(x,v,s) void0(@() Fl.W.Dtb.Drift.plot_drift_by_cond)
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_tnd)
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_rt_distrib( ...
                'd_cond', 1))
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_rt_distrib( ...
                'd_cond', 2))
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_rt_distrib( ...
                'd_cond', 3))
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_rt_distrib( ...
                'd_cond', 4))
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_rt_distrib( ...
                'd_cond', 5))
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_rt_distrib( ...
                'd_cond', 6))
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_rt_distrib( ...
                'd_cond', 7))
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_rt_distrib( ...
                'd_cond', 8))
            @(Fl) @(x,v,s) void0(@() Fl.W.plot_rt_distrib( ...
                'd_cond', 9))
%             @(Fl) @(x,v,s) void0(@() Fl.W.plot_rt_distrib( ...
%                 'd_cond', round(numel(Fl.W.Data.conds{1}) / 2)))
            });        
    end
    function plot_plotfuns(W)
        Fl = W.get_Fl;
        if isempty(Fl.PlotFcns)
            W.add_plotfun(Fl);
        end
        Fl.runPlotFcns;
    end
end
%% Plot
methods
    function plot_and_save_all(W)
        if isempty(W.Fl) || ~W.Fl.is_valid_res
            warning('No valid res!');
            return;
        end
        
        for kind = {'rt', 'ch', 'bound'}
            file = W.get_file({'plt', kind{1}});
            
            if exist(file, 'file') && W.skip_existing_fig
                fprintf('Skipping existing fig %s: %s\n', kind{1}, file);
            else
                if ~W.Data.is_pred_done
                    W.pred;
                end
                
                clf;
                W.(['plot_' kind{1}]);
                savefigs(file);
            end
        end
        
        kind{1} = 'plotfuns';
        file = W.get_file({'plt', kind{1}});
        if exist(file, 'file') && ~W.skip_existing_fig
            fprintf('Skipping existing fig %s: %s\n', kind{1}, file);
        else
            if ~W.Data.is_pred_done
                W.pred;
            end
            
            clf;
            Fl = W.get_Fl;
            Fl.runPlotFcns;
            savefigs(file, 'size', [1200, 900]);
        end
    end
    function [h_data, h_pred, Pl_data, Pl_pred] = plot_ch(W, varargin)
        Pl_pred = DtbPlot.PlotCh1D( ...
            W.Data.get_RT_pred_pdf, {
                'conds', W.Data.conds{1}
                'dt', W.get_dt
                'foldAxis', [false, true]
                }, {
                'Marker', 'none'
                });
        h_pred = Pl_pred.plot;
        hold on;
        
        Pl_data = DtbPlot.PlotCh1D( ...
            W.Data.get_RT_data_pdf, {
                'conds', W.Data.conds{1}
                'dt', W.get_dt
                'foldAxis', [false, true]
                }, {
                'Marker', 'o'
                'LineStyle', 'none'
                });
        h_data = Pl_data.plot;
        hold off;
    end
    function [h_data, h_pred, Pl_data, Pl_pred] = plot_rt(W, varargin)
        S = varargin2S(varargin, {
            'y_fun', 'mean'
            'thres_n_tr', 5
            });
        
        Pl_pred = DtbPlot.PlotRt1D( ...
            W.Data.get_RT_pred_pdf, varargin2C(S, {
                'conds', W.Data.conds{1}
                'dt', W.get_dt
                'foldAxis', [false, true]
                'x_bias', W.Dtb.Drift.th.bias
                }), {
                'Marker', 'none'
                });
        [h_pred.correct, h_pred.wrong] = Pl_pred.plot;
        hold on;
        
        Pl_data = DtbPlot.PlotRt1D( ...
            W.Data.get_RT_data_pdf, varargin2C(S, {
                'conds', W.Data.conds{1}
                'dt', W.get_dt
                'foldAxis', [false, true]
                'x_bias', W.Dtb.Drift.th.bias
                }), {
                'Marker', 'o'
                'LineStyle', 'none'
                });
        [h_data.correct, h_data.wrong] = Pl_data.plot;
        hold off;
        
%         if strcmp(S.y_fun, 'var')
            %%
            n = squeeze(sum(Pl_data.pdf_permuted));
            excl = n < S.thres_n_tr;
            
            for ch = 1:2
                h_data.wrong(ch).YData(excl(:,ch)) = nan;
                h_pred.wrong(ch).YData(excl(:,ch)) = nan;
            end
            
            y_vec = [[h_data.correct.YData], ...
                     [h_data.wrong.YData], ...
                     [h_pred.correct.YData], ...
                     [h_pred.wrong.YData]];
            min_y = nanmin(y_vec);
            max_y = nanmax(y_vec);
            range_y = max_y - min_y;
            
            ylim([max_y - range_y * 1.05, ...
                  min_y + range_y * 1.05]);
            
%         end
    end
    function [h_bound, h_drift] = plot_bound(W, varargin)
        h_bound = W.Dtb.Bound.plot;
        hold on;
        h_drift = W.Dtb.Drift.plot_drift_cond_t;
        hold off;
        bound = W.Dtb.Bound.get_bound_t_ch;
        ylim([min(bound(:)) - 0.1, max(bound(:)) + 0.1]);
        xlim(W.t([1, end]));
    end
    function h = plot_tnd(W, varargin)
        h = W.Tnd.plot;
    end
    function h = plot_rt_distrib(W, varargin)
        S = varargin2S(varargin, {
            'p_pred', W.Data.RT_pred_pdf
            'p_data', W.Data.RT_data_pdf
            'd_cond', 1
            });
        
        ps = {S.p_pred, S.p_data};
        specs = {'r-', 'k-'};
        
        t = W.t;
        h = ghandles(2,2);
        
        for ii = 1:2
            p = ps{ii};
            p = permute(p(:, S.d_cond, :), [1, 3, 2]);
            p_max = max(p(:));
            p = nan0(p ./ p_max);
            
            for ch = 1:2
                sgn = sign(ch - 1.5);
                
                h(ii,ch) = plot(t, p(:,ch) .* sgn, specs{ii}, ...
                    'LineWidth', 1);
                hold on;
            end
        end
        hold off;
        crossLine('h', 0);
        xlabel('Time (s)');
        bml.plot.beautify;
    end
end
%% Constrain Tnd
methods
    constrain_tnd_from_data(W)
end
%% Demo
methods (Static)
    function varargout = demo(varargin)
        [varargout{1:nargout}] = Fit.D1.Bounded.Main.fit_unit(varargin{:});
    end    
    function [Fl, S] = demo_fit_bound_from_zero(varargin)
        C = varargin2C(varargin, {
            'adCond_incl', 1
            'Drift', 'Indiv'
            });
        Fl0 = Fit.D1.Bounded.Main.fit_unit(C{:});
        assignin('base', 'Fl0', Fl0); % DEBUG

        %%
        W_th0 = deep_copy(Fl0.W);

        W_th0.Dtb.Bound.fix_to_th_;
        W_th0.Tnd.fix_to_th_;
        W_th0.Miss.fix_to_th_;

        %%
        C = varargin2C(varargin, {
            'Drift', 'Indiv'
            'W_th0', W_th0 % DEBUG
            'fix_params', true % DEBUG
            });
        [Fl, S] = Fit.D1.Bounded.Main.fit_unit(C{:});    
    end
end
end