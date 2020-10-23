classdef PlotRt1D < DtbPlot.PlotPdf1D
    % DtbPlot.PlotRt1D
    %
    % 2015 YK wrote the initial version.
        
properties
    y_fun = 'mean';
    to_plot_wrong = true;
end
    
methods
    function Pl = PlotRt1D(varargin)
        % Pl = PlotRt1D(pdf, varargin)
        % pdf(t, cond, ch) : probability mass

        Pl = Pl@DtbPlot.PlotPdf1D(varargin{:});
        if nargout == 0
            Pl.plot;
        end
    end
    function [h_correct, h_wrong] = plot(Pl, varargin)
        y = Pl.get_y;
        x = Pl.get_x;
        ax = Pl.ax;
        C = bml.plot.varargin2plot(varargin, ...
                bml.plot.varargin2plot(Pl.plotArgs, {
                    'Color', 'k'
                    'MarkerFaceColor', 'k'
                    'MarkerEdgeColor', 'w'
                }));
            
        n_ch = 2;
        x_bias = x - Pl.x_bias;
        for ch = n_ch:-1:1
            incl = sign(ch - 1.5) ~= -sign(x_bias);
            
            x1 = x;
            y1 = y(:, ch);
            x1(~incl) = nan;
            y1(~incl) = nan;
            h_correct(ch) = plot(ax, x1, y1, C{:});
            
            hold on;
        end
        
        if Pl.to_plot_wrong
            C1 = bml.plot.varargin2plot({
                    'Color', 'r'
                    'MarkerFaceColor', 'r'
                    }, C);
            for ch = n_ch:-1:1
                incl = sign(ch - 1.5) == -sign(x_bias);
                
                x1 = x;
                y1 = y(:, ch);
                x1(~incl) = nan;
                y1(~incl) = nan;
                h_wrong(ch) = plot(ax, x1, y1, C1{:});
                hold on;
            end
        else
            h_wrong = gobjects(1, n_ch);
        end
        hold off;
        
        bml.plot.lim_margin('axis', 'y', 'direction', 'pos');

        [x_tick, x_ticklabel] = Pl.get_x_tick;
        if ~isempty(x_tick)
            set(gca, ...
                'XTick', x_tick, ...
                'XTickLabel', csprintf('%g', x_ticklabel));
        end

        if Pl.foldDim(Pl.dimOnX)
            if ~isempty(x_tick)
                xlim([x_tick(1), ...
                    x_tick(1) + (max(x_tick) - min(x_tick)) * 1.05]);
            end
        else
            max_abs_x = max(abs(Pl.x));
            xlim([-1.05, 1.05] .* max_abs_x);
        end

        bml.plot.beautify;        
    end
    function x = get_x(Pl)
        x = Pl.conds;
    end
    function y = get_y(Pl) 
        p = Pl.pdf;

        switch Pl.y_fun
            case 'mean'
                y = mean_distrib(p, Pl.t(:), 1);
            case 'var'
                y = std_distrib(p, Pl.t(:), 1).^2;
        end

        y = squeeze(y);
        if Pl.dimOnX == 2
            y = y';
        end
        if isvector(y)
            y = y(:);
        end    
        
%         [~, y] = Pl.get_x(y);
    end
    function set_y_fun(Pl, y_fun)
        assert(ischar(y_fun) && any(strcmp(y_fun, {'mean', 'var'})));
        Pl.y_fun = y_fun;
    end
end
methods (Static)
    function stop = outputfun(~, ~, ~, varargin)
        stop = false;
        Pl = DtbPlot.PlotRt1D(varargin{:});
        Pl.plot;
    end
end
end % classdef