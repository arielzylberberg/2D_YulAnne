classdef UnabsWithTd2D < FitWorkspace
properties
    Fl = FitFlow;
    
    % cond_plot(dim)
    % : A joint condition to draw. Defaults to an intermediate level.
    cond_plot = [8 8]; 

    dim_plot = 1;
    t_plot_max = 1.5;
    
    t_of_interest
end
properties (Dependent)
    W
    Dtb
    bound
    unabs_together % (t, y) of the cond_plot(dim_plot)
    td_together % (t, ch) of the cond_plot(dim_plot)
    td_together_first % (t, ch) of the cond_plot(dim_plot)
    t_plot
    t_incl
    y_plot
    y_incl
end
methods
    function Il = UnabsWithTd2D(varargin)
        if nargin > 0
            Il.init(varargin{:});
        end
    end
    function init(Il, Fl, varargin)
        varargin2props(Il, varargin);
        
        if exist('Fl', 'var') && ~isempty(Fl)
            Il.Fl = Fl;
            Il.W.pred;
        end
    end
    function plot(Il)
        Il.imagesc_unabs_together;
        hold on;
        
        y_plots1 = Il.area_td_together;
        hold on;
        
        y_plots2 = Il.area_td_together_first;
        hold on;
        
        y_plots_all = [y_plots1(:); y_plots2(:)];
        ylim([min(y_plots_all), max(y_plots_all)]);
        
        axis off;
    end
    function imagesc_unabs_together(Il)
        y_plot = Il.y_plot;
        t_plot = Il.t_plot;
        
        unabs = Il.unabs_together;
             
        imagesc(t_plot, y_plot, unabs');
        set(gca, 'CLim', [min(unabs(:)), max(unabs(:))]);
        
        axis xy;
        colormap(parula(256));
        bml.plot.beautify;
    end
    function varargout = area_td_together(Il)
        [varargout{1:nargout}] = ...
            Il.area_td(Il.td_together, 'k');
    end
    function varargout = area_td_together_first(Il)
        [varargout{1:nargout}] = ...
            Il.area_td(Il.td_together_first, lines(1));
    end
    function y_plots = area_td(Il, td, color)
        t_plot = Il.t_plot;
        y_plot = Il.y_plot;
        dy = y_plot(2) - y_plot(1);
        
        td = td ./ max(td(:));
        
        base = hVec(y_plot([1, end])) + dy .* [-1, 1]./2;
        fac = base .* 0.4;
        
        for ch = 2:-1:1
            td_plot = td(:,ch) * fac(ch) + base(ch);

            patch( ...
                'XData', t_plot([1, 1:end, end]), ...
                'YData', [base(ch); td_plot; base(ch)], ...
                'FaceColor', color, ...
                'FaceAlpha', 0.25, ...
                'EdgeColor', 'none');
            hold on;
            
            plot(t_plot, td_plot, 'Color', color);
            hold on;
            
            y_plots(:,ch) = td_plot;
        end
        hold off;
    end
end
%% Get/Set
methods
    function v = get.W(Il)
        v = Il.Fl.W;
    end
    function v = get.Dtb(Il)
        v = Il.W.Dtb;
    end
    function v = get.bound(Il)
        Bound = Il.Dtb.Bounds{Il.dim_plot};
        v = Bound.get_bound_t_ch;
    end
    function v = get.unabs_together(Il)
        v = Il.Dtb.unabs_together{Il.dim_plot} ...
            (Il.t_incl, Il.y_incl, Il.cond_plot(Il.dim_plot));
    end
    function v = get.td_together(Il)
        for ch = 2:-1:1
            v(:,ch) = Il.Dtb.td_together{Il.dim_plot} ...
                (Il.t_incl, Il.cond_plot(Il.dim_plot), ch);
        end
    end
    function v = get.td_together_first(Il)
        for ch = 2:-1:1
            v(:,ch) = Il.Dtb.td_together_first{Il.dim_plot} ...
                (Il.t_incl, Il.cond_plot(Il.dim_plot), ch);
        end
    end
    function y_plot = get.y_plot(Il)
        y_all = Il.Dtb.y;        
        y_plot = y_all(Il.y_incl);
    end
    function y_incl = get.y_incl(Il)
        y_all = Il.Dtb.y;
        bound = Il.bound;
        max_y = [min(bound(:,1)), max(bound(:,2))];
        max_y_ix = [find(y_all > max_y(1), 1, 'first'), ...
                 find(y_all < max_y(2), 1, 'last')];
        y_incl = max_y_ix(1):max_y_ix(2);
    end
    function t_plot = get.t_plot(Il)
        t_all = Il.Dtb.t;
        t_plot = t_all(Il.t_incl);
    end
    function t_incl = get.t_incl(Il)
        t_all = Il.Dtb.t;
        t_incl = t_all <= Il.t_plot_max;
    end
end
end