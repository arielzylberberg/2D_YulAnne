classdef PlotMetaVarDifficulty < DtbPlot.PlotMetaDifficulty
methods
    function Pl = PlotMetaVarDifficulty(varargin)
        Pl = Pl@DtbPlot.PlotMetaDifficulty(varargin{:});
        Pl.componentClass = 'PlotRt2D';
        Pl.componentArgs  = {{'y_fun', 'var'}};
    end
end
end