classdef Main < Fit.D2.Bounded.Main
    % Fit.D2.RT.BoundedCondEn.Main
    %
    % Same as Bounded.Main except for Data and Dtb.
    % Uses Td/RT_pred/data_pdf_tr (note '_tr').
    %
    % 2015 YK wrote the initial version.
methods
    %% Fit
    function pred(W)
        W.Dtb.pred;
        W.Data.set_RT_pred_pdf_tr(W.Tnd.Td2RT(W.Data.get_Td_pred_pdf_tr));
        W.Data.set_RT_pred_pdf_tr(W.Miss.add_miss(W.Data.get_RT_pred_pdf_tr));
    end
    function [c, c_sep] = calc_cost(W)
        [c, c_sep] = nll_bin( ...
            W.Data.get_RT_pred_pdf_tr, ...
            W.Data.get_RT_data_pdf_tr);
    end
    %% Set object properties
    function set_Data(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        obj_or_name = ...
            W.enforce_class('Fit.D2.Common.DataChRtPdfEn', ...
                obj_or_name);
        W.set_Data@Fit.D2.Bounded.Main(obj_or_name);
    end
    function set_Dtb(W, obj_or_name)
        if nargin < 2, obj_or_name = 'WithDtbCalc'; end
        W.Dtb = W.enforce_class('Fit.D2.RT.BoundedCondEn.Dtb', obj_or_name);
        W.set_sub_from_props({'Dtb'});
    end
end
methods (Static)
    function W = demo(varargin)
        S = varargin2S(varargin, {
            'DriftEn', 'Mean' % 'Mean' or ''
            });
        
        %% Initialize
%         W = eval(my_class);
        W = Fit.D2.RT.BoundedCondEn.Main;
        
        %% Customize
        W.Dtb.Drift.set_Drifts(repmat({S.DriftEn}, [1, 2]));
        
        %%
        W.Data.set_path;
        W.Data.load_data;
        
        %%
        W.Dtb.DtbCalc.set_n_rep_DtbCalc(10); % For testing
        tic; W.pred; toc;
        
        %% Gather output
        p = W.Data.get_RT_pred_pdf;
        
        %% Plot Ch
        subplot(2,1,1);
        Pl = DtbPlot.PlotCh2D(p);
        Pl.plot;
        
        %% Plot RT
        subplot(2,1,2);
        Pl = DtbPlot.PlotRt2D(p);
        Pl.plot;
        
        %% Fix unnecessary params
        W.th.Dtb__p_dim1_1st = 0.5;
        W.th.Dtb__sigmaSq_fac_together_dim1_1 = 0;
        W.th.Dtb__sigmaSq_fac_together_dim1_2 = 0;
        W.th.Dtb__sigmaSq_fac_together_dim2_1 = 0;
        W.th.Dtb__sigmaSq_fac_together_dim2_2 = 0;
        W.th.Dtb__TndSt__mu = 0.05;
        W.th.Dtb__TndSt__disper = W.lb.Dtb__TndSt__disper;
        W.th.Miss__miss = 0.001;
        W.fix_to_th_({
            'Dtb__p_dim1_1st'
            'Dtb__Drift__Drift1__bias'
            'Dtb__Drift__Drift2__bias'
            'Dtb__Bound__Bound1__bias'
            'Dtb__Bound__Bound2__bias'
            'Dtb__sigmaSq_fac_together_dim1_1'
            'Dtb__sigmaSq_fac_together_dim1_2'
            'Dtb__sigmaSq_fac_together_dim2_1'
            'Dtb__sigmaSq_fac_together_dim2_2'
            'Dtb__TndSt__mu'
            'Dtb__TndSt__disper'
            'Miss__miss'
            });
        
        %% Give a good starting point
        W.th0.Dtb__Drift__Drift1__k = 30;
        W.th0.Dtb__Drift__Drift2__k = 5;
        
        %% Prepare fit
        Fl = W.get_Fl;
        
        %% Add plotfun
        Fl.remove_plotfun_all;
        W.add_plotfun(Fl);
        
        %% Test run
        Fl.W.pred;
        
        %%
        Fl.runPlotFcns;
        
        %% Fit
        Fl.fit('opts', {'UseParallel', 'always'});
%         Fl.fit('opts', {'MaxIter', 1});
        
        %% Plot results
        W = Fl.W;
    end
end
end