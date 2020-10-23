classdef PlotMetaVarEasiest < DtbPlot.PlotMetaEasiest
methods
    function Pl = PlotMetaVarEasiest(varargin)
        Pl = Pl@DtbPlot.PlotMetaEasiest(varargin{:});
        Pl.componentClass = 'PlotRt2D';
        Pl.componentArgs  = {{'y_fun', 'var'}};
    end
end
end