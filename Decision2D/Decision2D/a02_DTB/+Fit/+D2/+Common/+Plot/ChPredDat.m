classdef ChPredDat < Fit.D2.Common.Plot.Adaptor
properties
    PlPred
    PlData
end
methods
    function Plt = ChPredDat(varargin)
        Plt.init_Pls0;
        if nargin > 0
            Plt.init(varargin{:});
            Plt.plot;
        end
    end
    function init_Pls0(Plt)
        Plt.PlPred = Fit.D2.Common.Plot.Ch;
        Plt.PlData = Fit.D2.Common.Plot.Ch;
    end
    function init(Plt, W, varargin)
        if ~exist('W', 'var')
            W = [];
        end
        
        opt_pred = varargin2C({
            'pdf_kind', 'RT_pred_pdf'
            }, varargin);
        Plt.PlPred.init(W, opt_pred{:});
        
        opt_dat = varargin2C({
            'pdf_kind', 'RT_data_pdf'
            }, varargin);
        Plt.PlData.init(W, opt_dat{:});
    end
    function plot(Plt, varargin)
        Plt.PlPred.plot(varargin{:});
        hold on;
        Plt.PlData.plot(varargin{:});
        hold off;
    end
end
end
    