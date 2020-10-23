classdef PlotOverlayRtShortestRt2D < DtbPlot.PlotOverlay
methods
    function Pl = PlotOverlayRtShortestRt2D(varargin)
        Pl = Pl@DtbPlot.PlotOverlay(varargin{:});
        Pl.CompoPl = DtbPlot.PlotRtShortestRt2D;
    end
end
end