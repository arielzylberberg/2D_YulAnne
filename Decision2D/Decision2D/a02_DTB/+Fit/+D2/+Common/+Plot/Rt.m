classdef Rt < Fit.D2.Common.Plot.Ch
properties
    y_fun = 'mean';
end
methods
    function Plt = Rt(varargin)
        if nargin > 0
            Plt.init(varargin{:});
            Plt.plot;
        end
    end
end
methods (Hidden)
    function calc_Pl(Plt)
        if Plt.is_pred_pdf
            Pl_args = varargin2C(Plt.Pl_args, {
                'x_bias', Plt.x_bias
                });
        else
            Pl_args = varargin2C(Plt.Pl_args);
        end
        
        Plt.Pl = DtbPlot.PlotRt2D( ...
            Plt.p, ...
            varargin2C(Pl_args), ...
            varargin2C(Plt.plot_args));
    end
end
end