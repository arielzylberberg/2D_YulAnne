classdef LogisticSlope < Fit.D2.Common.Plot.Logistic
methods
    function Plt = LogisticSlope(varargin)
        if nargin > 0
            Plt.init(varargin{:});
            Plt.plot;
        end
    end
end
end