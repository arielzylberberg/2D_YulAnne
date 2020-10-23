classdef PlotCh1D < DtbPlot.PlotPdf1D
    % DtbPlot.PlotCh1D
    %
    % 2015 YK wrote the initial version.

methods
    function Pl = PlotCh1D(cPdf, plArgs, plotArgs)
        % Pl = PlotPdf1D(cPdf, plArgs, plotArgs)
        %
        % cPdf(t, cond1, ch1) : probability mass
        % plArgs: properties of Pl
        % plotArgs: arguments of plot()
        if nargin < 1, cPdf = []; end
        if nargin < 2, plArgs = {}; end
        if nargin < 3, plotArgs = {}; end

        if ~isempty(cPdf)
            Pl.set_pdf(cPdf);
        end
        varargin2fields(Pl, plArgs);
        Pl.plotArgs = plotArgs;
        if Pl.plotNow || (nargout == 0 && ~isempty(cPdf))
            Pl.plot(plotArgs{:});
        end
    end
    function x = get_x(Pl)
        x = Pl.conds;
    end
    function y = get_y(Pl)
    %     y = get_y@DtbPlot.PlotPdf2D(Pl);
    %     y = sum(y, 1);

%         dimSep = Pl.dimSep;

        y = sums(Pl.pdf, [1, 3 + Pl.dimSep], true); % marginalize choice over sepDim
        if size(Pl.pdf, 1 + Pl.dimOnX) == 1
            y = y'; % y should be a row vector in this case.
        end

        % No sep, unlike 2D

        %% Fold x
        if Pl.foldDim(Pl.dimOnX)
            if size(y,1) > 1
                mid1 = round(size(y,1)/2);
                mid2 = ceil((size(y,1)+1)/2);
                y = (y(mid2:end,[1 2]) + y(mid1:-1:1,[2 1])) / 2;    
            end

            %% Calculate proportion
            ySum = squeeze(sum(y,2));
            y1   = y(:,2);
            y    = y1 ./ ySum;    
        else
            %% Calculate proportion
            ySum = squeeze(sum(y,2));
            y1   = y(:,2);
            y    = y1 ./ ySum;    
        end
    end
    function h = plot(Pl, varargin)
        y = Pl.get_y;
        x = Pl.get_x;
        ax = Pl.ax;
        C = bml.plot.varargin2plot(varargin, ...
                bml.plot.varargin2plot(Pl.plotArgs, {
                    'Color', 'k'
                    'MarkerFaceColor', 'k'
                    'MarkerEdgeColor', 'w'
                }));
        h = plot(ax, x, y, C{:});
        
        [x_tick, x_ticklabel] = Pl.get_x_tick;
        if ~isempty(x_tick)
            set(gca, 'XTick', x_tick, 'XTickLabel', x_ticklabel);
        end
        
        if Pl.foldDim(Pl.dimOnX)
            ylim([0.5 1]);
            
            if ~isempty(x_tick)
                xlim([x_tick(1), ...
                    x_tick(1) + (max(x_tick) - min(x_tick)) * 1.05]);
            end
        else
            ylim([0 1]);
            
            max_abs_x = max(abs(Pl.x));
            xlim([-1.05, 1.05] .* max_abs_x);
        end
        set(gca, ...
            'YTick', 0:0.25:1, ...
            'YTickLabel', {'0', '', '0.5', '', '1'});
        
        bml.plot.beautify;
    end
end
methods (Static)
    function stop = outputfun(~, ~, ~, varargin)
        stop = false;
        Pl = DtbPlot.PlotCh1D(varargin{:});
        Pl.plot;
    end
end
end