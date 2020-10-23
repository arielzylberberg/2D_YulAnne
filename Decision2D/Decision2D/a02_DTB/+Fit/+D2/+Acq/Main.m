classdef Main < Fit.D2.Inh.MainBatch % & Fit.Common.Main
    % Fit.D2.Acq.Main
    %
    % 2016 YK wrote the initial version.
properties (Dependent)
    buffer_dur_sec % Reflects Main.Dtb's buffer duration
    to_fix_buffer_dur
end
%% BatchFit interface
methods
    function W = Main(varargin)
%         W.set_Data; % Just to make sure Data is shared.
        W.parad = 'sh';
%         W.set_dt(0.025); % Why is this?? Is this dispensible?
        
        W.buffer_dur_sec = 0.12;
        W.to_fix_buffer_dur = true;
        
        W.bound_kind = 'Const';
        W.sigmaSq_kind = 'Const';

        W.p_dim1_1st = 0.5;
        W.fix_p_dim1_1st = false;
        W.fix_bias_irr_1 = true;
        W.fix_bias_abs_irr_1 = true;
        W.fix_bias_irr_2 = true;
        W.fix_bias_abs_irr_2 = true;
        
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        S = varargin2S(varargin, {
            'buffer_dur_sec', W.buffer_dur_sec
            'to_fix_buffer_dur', W.to_fix_buffer_dur
            });
        
        W.init@Fit.D2.Inh.MainBatch(varargin{:});
        
        % Reapply cached props 
        % - prevents reset during init@Fit.D2.Inh.MainBatch.
        bml.oop.varargin2props(W, S, true);        
    end
    function S = get_S_fit_unit(~, varargin)
        S = varargin2S(varargin, {
            'subj', Data.Consts.subjs_short{1}
            'parad', 'sh'
            'task', 'A'
            ... % Dtb
            'to_fix_buffer_dur', false
            'buffer_dur_msec', 120
            'to_fix_p_dim1_1st', false
            'p_dim1_1st_100x', 50
            ...
            'k_b_prod_fac_10x', 20 % 10 * k_b_prod_fac (for file names)
            ...
            'Dtb', 'Serial'
            'Bound', 'Const'
            ... % fit opt
            'fit_args', {}
            ... % DEBUG
            'to_plot_plotfun', true
            });
    end
end
%% User interface
methods
    function Fl = demo_fit(W)
        %%
        W.set_max_t(2);
        W.set_dt(0.02);
        W.Dtb.init_Td_intermediate;
        
        W.Data.set_path({'subj', Data.Consts.subjs_short{1}, 'parad', 'sh'});
        W.Data.load_data;
        
%         k_b_prod_fac = 1.5;
%         KBRatio1 = Fit.D2.Common.KBRatio( ...
%             'k_b_prod_range', k_b_prod_fac);
%         KBRatio2 = deep_copy(KBRatio1);
%         W.Dtb.set_KBRatios({KBRatio1, KBRatio2});
        
        %%
        for buffer_dur_sec_ = 0.25
%             W.buffer_dur_sec = buffer_dur_sec_;
            
            Fl = W.get_Fl;
            W = Fl.W;
            
            W.Dtb.th0.buffer_dur_sec = buffer_dur_sec_;
            W.Dtb.th_lb.buffer_dur_sec = 0.06;
            W.Dtb.th_ub.buffer_dur_sec = 1;
            
%             W.Dtb.Drift1.th0.k = 10;
%             W.Dtb.Drift2.th0.k = 10;
%             W.Dtb.Drift1.th_ub.k = 30;
%             W.Dtb.Drift2.th_ub.k = 30;
%             
%             W.Dtb.Bound1.th0.b = 0.5;
%             W.Dtb.Bound2.th0.b = 0.5;
            
            W.Tnd.th0.mu = 0.2;
            W.Tnd.th_lb.mu = 0.1;
            W.Tnd.th_lb.mu_UmD = -0.1;
            W.Tnd.th_ub.mu_UmD = 0.1;
            W.Tnd.th_lb.mu_RmL = -0.1;
            W.Tnd.th_ub.mu_RmL = 0.1;

            W.Dtb.Drift1.fix_({'bias'});
            W.Dtb.Drift2.fix_({'bias'});
            
            Fl = W.demo_fit_Fl(Fl);
        end
    end
    function demo_cumsum_Td(W)
        %%
        W.Data.set_path;
        W.Data.load_data;
        W.Dtb.TO_DEBUG = 2; % DEBUG
        
        %%
        n_row = 3;
        n_col = 1;
        
        for buffer_dur_sec_ = [0.12, 0.2, 0.3, 0.5, 1]
            W.buffer_dur_sec = buffer_dur_sec_;
            
%             [td_pdf, unabs] = W.Dtb.get_Td_pdf;
            W.pred;
            pred_pdf = W.Data.get_RT_pred_pdf;
            
            %%
            subplotRC(n_row, n_col, 1, 1);
            f = @(p) cumsum(reshape(sums(p, [4, 5], true), size(p, 1), []));
            t = W.get_t;
            plot(t, f(pred_pdf));
            crossLine('v', buffer_dur_sec_);
            grid on;
            
            title(sprintf('buffer dur (s) : %1.3f', buffer_dur_sec_));
            
            %%
            subplotRC(n_row, n_col, 2, 1);
            Pl = DtbPlot.PlotRt2D(pred_pdf, {'dt', W.get_dt});
            Pl.plot;
            y_lim = ylim;
            ylim([0, y_lim(2)]);
            crossLine('h', buffer_dur_sec_);
            
            subplotRC(n_row, n_col, 3, 1);
            Pl = DtbPlot.PlotCh2D(pred_pdf, {'dt', W.get_dt});
            Pl.plot;
            ylim([0.5 1]);
            crossLine('h', buffer_dur_sec_);
            
            %%
            keyboard;
%             input('Press ENTER to continue:', 's');
        end
    end
end
%% Internal
methods
    function varargout = fit_unit(W, varargin)
        [varargout{1:nargout}] = W.fit_unit@Fit.Common.Main(varargin{:});
    end
    function set_Dtb(W, obj_or_name)
        if nargin < 2, obj_or_name = 'DensityIndivJt'; end
        W.Dtb = W.enforce_class('Fit.D2.Acq.Dtb', obj_or_name);
        W.set_sub_from_props({'Dtb'});
    end
    function v = get_Drift1(W)
        v = W.Dtb.Drift1;
    end
    function v = get_Drift2(W)
        v = W.Dtb.Drift2;
    end
    function v = get_Bound1(W)
        v = W.Dtb.Bound1;
    end
    function v = get_Bound2(W)
        v = W.Dtb.Bound2;
    end
    function v = get_SigmaSq1(W)
        v = W.Dtb.SigmaSq1;
    end
    function v = get_SigmaSq2(W)
        v = W.Dtb.SigmaSq2;
    end
end
methods
    function set.buffer_dur_sec(W, v)
        W.Dtb.th.buffer_dur_sec = v;
%         W.Dtb.set_buffer_dur_sec_fix(v); % Use for testing only.
    end
    function v = get.buffer_dur_sec(W)
        v = W.Dtb.th.buffer_dur_sec;
%         v = W.Dtb.buffer_dur_sec;
    end
    function v = get.to_fix_buffer_dur(W)
        v = W.Dtb.th_fix.buffer_dur_sec;
    end
    function set.to_fix_buffer_dur(W, v)
        W.Dtb.th_fix.buffer_dur_sec = v;
    end
end
end