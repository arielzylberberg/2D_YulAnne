classdef Plot < DeepCopyable
    % Fit.D2.Bounded.Plot
    %
    % Plotting methods all should take a struct, 'opt' only and 
    % use set_size at the end.
    %
    % 2016 YK wrote the initial version.
properties
    Fl
    conds_oversample_factor = 1;
end
methods
    function Plt = Plot(varargin)
        Plt.set_Fl(varargin{:});
    end
    function set_Fl(Plt, Fl)
        if ~exist('Fl', 'var') || isempty(Fl)
%             Main = eval(bml.pkg.get_class_rel(Plt, 'Main'));
%             Plt.Fl = Main.get_Fl;
            Plt.Fl = FitFlow;
        else
            Plt.Fl = Fl;
        end
    end
    function plotfuns(Plt, opt)
        Fl = Plt.Fl;
        
        Fl.remove_plotfun_all;
        Fit.D2.Common.Plot.PlotFuns.add_plotfun(Fl);
        Fl.plot_opt.to_plot = true;
        Fl.runPlotFcns;
        
        set_size(gcf, [1200 800]);
    end
    function rt(Plt, opt)
        if ~exist('opt', 'var'), opt = struct; end
        C = varargin2C(opt, {
            'dimOnX', 1
            'foldAxis', [false, true]
            'conds_oversample_factor', Plt.conds_oversample_factor
            });
        Fit.D2.Common.Plot.RtPredDat(Plt.Fl.W, C{:});
        
        set_size(gcf, [300 200]);
    end
    function rt_stdev(Plt, opt)
        if ~exist('opt', 'var'), opt = struct; end
        C = varargin2C(opt, {
            'dimOnX', 1
            'foldAxis', [false, true]
            'y_fun', 'stdev'
            'conds_oversample_factor', Plt.conds_oversample_factor
            });
        Fit.D2.Common.Plot.RtPredDat(Plt.Fl.W, C{:});
        
        set_size(gcf, [300 200]);
    end
    function rt_skew(Plt, opt)
        if ~exist('opt', 'var'), opt = struct; end
        C = varargin2C(opt, {
            'dimOnX', 1
            'foldAxis', [false, true]
            'y_fun', 'skew'
            'conds_oversample_factor', Plt.conds_oversample_factor
            });
        Fit.D2.Common.Plot.RtPredDat(Plt.Fl.W, C{:});
        
        set_size(gcf, [300 200]);
    end
    function ch(Plt, opt)
        if ~exist('opt', 'var'), opt = struct; end
        C = varargin2C(opt, {
            'dimOnX', 1
            'foldAxis', [false, true]
            'conds_oversample_factor', Plt.conds_oversample_factor
            });
        Fit.D2.Common.Plot.ChPredDat(Plt.Fl.W, C{:});
        
        set_size(gcf, [300 200]);
    end
    function rt_log(Plt, opt)
        if ~exist('opt', 'var'), opt = struct; end
        C = varargin2C(opt, {
            'dimOnX', 1
            'logAxis', [true, false]
            'conds_oversample_factor', Plt.conds_oversample_factor
            });
        Fit.D2.Common.Plot.RtPredDat(Plt.Fl.W, C{:});
        
        set_size(gcf, [300 200]);
    end
    function ch_log(Plt, opt)
        if ~exist('opt', 'var'), opt = struct; end
        C = varargin2C(opt, {
            'dimOnX', 1
            'logAxis', [true, false]
            'conds_oversample_factor', Plt.conds_oversample_factor
            });
        Fit.D2.Common.Plot.ChPredDat(Plt.Fl.W, C{:});
        
        set_size(gcf, [300 200]);
    end
    function rtdistrib(Plt, opt)
        if ~exist('opt', 'var'), opt = struct; end
        C = varargin2C(opt, {
            'dimOnX', 1
            'foldAxis', [false, true]
            'conds_oversample_factor', Plt.conds_oversample_factor
            });
        Fit.D2.Common.Plot.RtDistribPredDat(Plt.Fl.W, C{:});
        
        set_size(gcf, [300 200]);
    end
end
end