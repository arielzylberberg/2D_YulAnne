classdef PlotRtShortestRt2D < DtbPlot.PlotRt2D
properties
    groupDif = [1 1 1 2 2];
    drawCrossLine = true;
end 
methods
function Pl = PlotRtShortestRt2D(varargin)
    % Pl = PlotRtShortestRt2D(pdf, varargin)
    % pdf(t, cond1, cond2, ch1, ch2) : probability mass
    
    Pl = Pl@DtbPlot.PlotRt2D(varargin{:});
end
function y = get_y(Pl, ~)
    if ~isempty(Pl.groupDif)
        nGroup = max(Pl.groupDif);
        y0 = Pl.get_y@DtbPlot.PlotRt2D;
        y  = zeros(size(y0,1), nGroup);
        for iGroup = 1:nGroup
            inGroup = Pl.groupDif == iGroup;
            
            y(:,iGroup) = nanmean(y0(:,inGroup), 2);
        end
    end
end
function x = get_x(Pl, ~)
    y = Pl.get_y;
    x = y(:,end);
end
function h = plot(Pl, varargin)
    h = Pl.plot@DtbPlot.PlotRt2D(varargin{:});
    delete(h{2});
    axis auto;
    if Pl.drawCrossLine
        crossLine('NE', 0, 'k:');
    end
end
end % methods
end % classdef