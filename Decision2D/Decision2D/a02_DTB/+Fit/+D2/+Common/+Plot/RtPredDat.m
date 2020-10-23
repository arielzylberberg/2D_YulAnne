classdef RtPredDat < Fit.D2.Common.Plot.ChPredDat
properties
    y_fun = 'mean';
end
methods
    function Plt = RtPredDat(varargin)
        Plt.init_Pls0;
        if nargin > 0
            Plt.init(varargin{:});
            Plt.plot;
        end
    end
    function init_Pls0(Plt)
        Plt.PlPred = Fit.D2.Common.Plot.Rt;
        Plt.PlData = Fit.D2.Common.Plot.Rt;        
    end
end
end