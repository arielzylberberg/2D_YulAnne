classdef Ch < Fit.D2.Common.Plot.Adaptor
    % Fit.D2.Common.Plot.Ch
    %
    % 2016 YK wrote the initial version.
properties
    h = [];
    
    plot_args = {};
    Pl_args = {};
    
    foldAxis = [true, true];
    logAxis = [false, false];
    conds_oversample_factor = 1; % 10; % DEBUG
end
%% Properties directly used for plotting
properties
    x = [];
    x_tick = [];
end
%% Objects
properties (Transient)
    Pl = [];
    W = [];
end
%% Plotting
methods
    function Plt = Ch(varargin)
        Plt.pdf_kind = 'RT_pred_pdf';
        if nargin > 0
            Plt.init(varargin{:});
            Plt.plot;
        end
    end
    function plot(Plt, varargin)
        Plt.h = Plt.Pl.plot(varargin{:});
    end
end
%% Internal - Beautification
methods
end
%% Internal - Calculation
methods (Hidden)
    function init(Plt, varargin)
        Plt.init@Fit.D2.Common.Plot.Adaptor(varargin{:});
        
        Plt.calc_plot_args_with_pdf_kind;
        
        % p, x
        Plt.calc_xy_with_pdf_kind;
        
        % Pl
        Plt.calc_Pl_args;
        Plt.calc_Pl;
    end
    function calc_plot_args_with_pdf_kind(Plt)
        if Plt.is_pred_pdf
            Plt.plot_args = varargin2C(Plt.plot_args, {
                'Marker', 'none'
                'LineStyle', '-'
                });
        else
            Plt.plot_args = varargin2C(Plt.plot_args, {
                'Marker', 'o'
                'LineStyle', 'none'
                'MarkerEdgeColor', 'w'
                });
        end                    
    end
    function calc_xy_with_pdf_kind(Plt)
        N_DIM = 2;
        
        to_oversample = Plt.is_pred_pdf;
        
        fac0 = Plt.W.Data.get_conds_oversample_factor;
        fac = ones(1, N_DIM);
        if to_oversample
            fac(Plt.dimOnX) = Plt.conds_oversample_factor;
            fac(Plt.get_dimSep) = 1;
        end
        if ~isequal(fac0, fac)
            Plt.W.Data.set_conds_oversample_factor(fac);
            if Plt.is_pred_pdf
                Plt.W.pred;
            end
        end
        
        Plt.calc_p;
        Plt.calc_x;

%         if to_oversample
%             Plt.Data.set_conds_oversample_factor(fac0);
%         end
    end
    function calc_p(Plt)
        Plt.p = Plt.any2p(Plt.W, Plt.pdf_kind);
    end
    function calc_x(Plt)
        Plt.x = Plt.W.Data.get_conds;
        Plt.x_tick = Plt.W.Data.get_conds_wo_oversample;
    end
    function calc_Pl_args(Plt)
        if isprop(Plt, 'y_fun')
            y_fun = Plt.y_fun;
        else
            y_fun = '';
        end
        
        Plt.Pl_args = varargin2S({
            'conds', Plt.x
            'conds_tick', Plt.x_tick
            'dimOnX', Plt.dimOnX
            'foldAxis', Plt.foldAxis
            'logAxis', Plt.logAxis
            'dt', Plt.W.dt
            'y_fun', y_fun
            }, Plt.Pl_args);
    end
    function calc_Pl(Plt)
        Plt.Pl = DtbPlot.PlotCh2D( ...
            Plt.p, ...
            varargin2C(Plt.Pl_args), ...
            varargin2C(Plt.plot_args));
    end
    function get_Pl_wo_plotting(Plt)
        Pl_args = varargin2C({
            'plotNow', false
            }, Plt.Pl_args);
        
        Plt.Pl = DtbPlot.PlotCh2D( ...
            Plt.p, ...
            Pl_args, ...
            Plt.plot_args);
    end
    function tf = is_pred_pdf(Plt)
        tf = Plt.is_pred(Plt.pdf_kind);
    end
end
methods (Static)
    function stop = plotfun(~,~,~, varargin) % x,v,s
        stop = false;
        Fit.D2.Common.Plot.Ch.(varargin{:});
    end
end
end
    