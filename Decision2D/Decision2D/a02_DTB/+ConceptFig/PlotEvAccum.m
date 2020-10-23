classdef PlotEvAccum < matlab.mixin.Copyable
properties
    dt = 0.1;
    slope0 = 1;
    thres0 = 1.2;
    arrow_style = {
        'TipAngle', 20, ...
        'BaseAngle', 45, ...
        'LineWidth', 2, ...
        'Length', 1}; % 5};
    xy0 = [0, 0];
    y_lim = [-0.6, 2.1 + 1.6];
end
methods
    function h = plot_accum(Pl, varargin)
        S = varargin2S(varargin, {
            'plot_arrow', true
            'height_ch', 0.3
            });
        
        %%
        cla;
        axis equal
        
        rng(5);
%         dt = Pl.dt;
        dt = 0.03;
        t = 0:dt:2;
        nt = length(t);
        x_lim = [-0.1, 1.5];
        slope = 0.5;
        thres = 1;
%         slope0 = Pl.slope0;
%         thres0 = Pl.thres0;
        
        n_sim = 5;
        margin_y = S.height_ch;
        
        mu = dt * slope;
        sig = sqrt(dt);
        
        xlim(x_lim);
        ylim((1 + margin_y) * [-1, 1] * thres);
        
        hold on;
        color_trace = 0.7 + [0, 0, 0];
%         thress = [-1, 1] * thres
        for i_sim = 1:n_sim
            dy = normrnd(mu, sig, [1, nt]);
            cy = cumsum(dy);
            td = find(abs(cy) >= thres, 1, 'first');
            ch = sign(cy(td));
            cy(td:end) = ch * thres;
            cy = [0, cy(2:end)];
            plot(t, cy, '-', 'Color', color_trace, ...
                'LineWidth', 0.5);
        end
        
        dt_arrow = 0.2;
        y = -thres:0.01:thres;
        p = normpdf(y, slope * dt_arrow, sqrt(dt_arrow));
        p = p / max(p) * 0.2;
        patch(p + dt_arrow, y, 0.7 + [0, 0, 0], ...
            'EdgeColor', 'none');

        if S.plot_arrow
            xy1 = [1, slope] * dt_arrow;
            arrow([0, 0], xy1 * 0.9, ...
                'Color', 'k', ...
                Pl.arrow_style{:});
    %         text(xy1(1)+0.1, xy1(2), '$$(dt,{\mu}dt)$$', ...
    %             'Interpreter', 'latex', ...
    %             'FontSize', 13);
        end
        
        res = dtb.pred.analytic_dtb(slope, t, thres, -thres, 0);
        p = res.up.pdf_t;
        p = p / max(p) * S.height_ch * res.up.p;
        patch(t([1,1:end,end]), thres + [0, p(:)', 0], 0.7 + [0, 0, 0], ...
            'EdgeColor', 'none');

        p = res.lo.pdf_t;
        p = p / max(p) * S.height_ch * res.lo.p;
        patch(t([1,1:end,end]), -thres - [0, p(:)', 0], 0.7 + [0, 0, 0], ...
            'EdgeColor', 'none');

        color_thres = 0 + [0, 0, 0];
        plot(t([1, end]), [0, 0] + thres, '-', 'Color', color_thres, ...
            'LineWidth', 1);
        hold on;
        plot(t([1, end]), [0, 0] - thres, '-', 'Color', color_thres, ...
            'LineWidth', 1);
        
        set(gca, 'XTick', [], 'YTick', []);
        xax = get(gca, 'XAxis');
        set(xax, 'Color', 'w');
        yax = get(gca, 'YAxis');
        set(yax, 'Color', 'w', 'Visible', 'on');
        h.ylabel = ylabel(sprintf('Accumulated evidence'));
        set(h.ylabel, 'Color', 'k');

        h.text = text(mean(x_lim), -thres - 0.1, 'Time', ...
            'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'top', ...
            'FontSize', 11);
%         h = xlabel('Time');
%         set(h, 'Color', 'k');
    end
    function [t0, t1] = plot_parallel(Pl, slope, color)
        %%
        x0 = Pl.xy0(1);
        y0 = Pl.xy0(2);
        
        t = 0:Pl.dt:2.3;
        slope0 = Pl.slope0;
        thres0 = Pl.thres0;

        thres = 1;
        bar_margin = 0.2;
        bar_height = 0.2;

        line_width = 2;

        t0 = thres0/slope0;
        t1 = thres/slope;
        plot(x0 + t, y0 + min(slope0 * t, thres0), '-', ...
            'Color', 'k', 'LineWidth', line_width);
        hold on;
        plot(x0 + t, y0 + min(slope * t, thres), '-', ...
            'Color', color, 'LineWidth', line_width);

        Pl.patch(x0 + [0, t1], y0 + [-2, -1] * bar_height - bar_margin, color);
        Pl.patch(x0 + [0, t0], ...
            y0 + [-1, 0] * bar_height - bar_margin, 'k');

        hold off;
        bml.plot.beautify;
        set(get(gca, 'XAxis'), 'visible', 'off');
        set(get(gca, 'YAxis'), 'visible', 'off');
        
%         arrow([

        % set(gca, 'YTick', []);
        xlim([-0.1, t(end)]);
%         ylim([-2 * bar_height - bar_margin, thres0 + 0.1]);
    end
    function [t0, t1] = plot_serial(Pl, slope, color)
        %%
        x0 = Pl.xy0(1);
        y0 = Pl.xy0(2);
        
        t = 0:Pl.dt:3.5;
        slope0 = Pl.slope0;
        thres0 = Pl.thres0;

        thres = 1;
        bar_margin = 0.2;
        bar_height = 0.2;

        line_width = 2;

        plot(x0 + t, y0 + min(slope0 * t, thres0), '-', ...
            'Color', 'k', 'LineWidth', line_width);
        hold on;

        t0 = thres0 / slope0;
        t1 = thres / slope + t0;
        plot(x0 + t, y0 + max(0, min(slope * (t - t0), thres)), '-', ...
            'Color', color, 'LineWidth', line_width);

        Pl.patch(x0 + [t0, t0 + thres/slope], ...
            y0 + [-1, 0] * bar_height - bar_margin, color);
        Pl.patch(x0 + [0, thres0/slope0], ...
            y0 + [-1, 0] * bar_height - bar_margin, 'k');

        hold off;
        bml.plot.beautify;
        set(get(gca, 'XAxis'), 'visible', 'off');
        set(get(gca, 'YAxis'), 'visible', 'off');

        % set(gca, 'YTick', []);
        xlim([-0.1, t(end)]);
%         ylim([-2 * bar_height - bar_margin, thres0 + 0.1]);
    end
    function plot_serial_switch(Pl, slope1, color)
        %%
        x0 = Pl.xy0(1);
        y0 = Pl.xy0(2);
        
        dt = Pl.dt;
        it = 1:57;
        it2t = @(it) (it - 1) * dt;
        t = it2t(it);
        nt = length(it);
        slope0 = Pl.slope0;
        thres0 = Pl.thres0;
        
        n_dim = 2;
        
        slope = [slope0, slope1];
        thres = [thres0, 1];
        y = zeros(1, n_dim);
        ys = zeros(nt, n_dim);
        
        t_mark = {};
        
        t_turn = [6, 5]; % [0.5, 0.5];
        t_switch = [3, 7]; % [0.1, 0.2];

        bar_margin = 0.2;
        bar_height = 0.2;

        line_width = 2;

        t1 = 1;
        turn = 2;
        oturn = 1;
        accum = 1;
        t_in_accum = 0;
        t_in_switch = 0;
        
        for it = 2:nt
            if accum
                y(turn) = min(y(turn) + dt * slope(turn), thres(turn));
                
                t_in_accum = t_in_accum + 1;
                if (t_in_accum >= t_turn(turn) ...
                        && y(oturn) < thres(oturn)) ...
                        || y(turn) >= thres(turn)
                    
                    t_mark{end+1} = {
                        turn, accum, it - t_in_accum, it
                        };
                    
                    t_in_accum = 0;
                    accum = 0;
                end
            else
                t_in_switch = t_in_switch + 1;
                if t_in_switch >= t_switch(turn)
                    t_mark{end+1} = {
                        turn, accum, it - t_in_switch, it
                        };
                    
                    t_in_switch = 0;
                    accum = 1;
                    turn = mod(turn, n_dim) + 1;
                    oturn = mod(oturn, n_dim) + 1;
                end
            end
            if all(y >= thres)
                for dim = 1:n_dim
                    ys(it:end, dim) = y(dim);
                end
                break;
            else
                for dim = 1:n_dim
                    ys(it, dim) = y(dim);
                end
            end
%             plot(ys);
        end
        
        plot(x0 + t, y0 + ys(:,1), '-', ...
            'Color', 'k', 'LineWidth', line_width);
        hold on;

        plot(x0 + t, y0 + ys(:,2), '-', ...
            'Color', color, 'LineWidth', line_width);

        colors = {[0,0,0], color};
        
        for ii = 1:numel(t_mark)
            [turn, accum, t_st, t_en] = deal(t_mark{ii}{:});
            
            if accum
                C = {colors{turn}};
            else
                C = {colors{turn}, 'FaceAlpha', 0.35};
                
                [~,~,t_st1,~] = deal(t_mark{ii-1}{:});
                h = plot(x0 + it2t([t_st1, t_en]), y0 + ys([t_st1, t_en], turn), ...
                    '-', 'Color', colors{turn}*0.4 + 0.6, ...
                    'LineWidth', line_width/2);
                uistack(h, 'bottom');
                
                h = plot(x0 + it2t([t_en, t_en]), y0 + [-bar_margin, ys(t_en, turn)], ...
                    ':', 'Color', colors{turn}*0.2 + 0.8, ...
                    'LineWidth', line_width/2);
                uistack(h, 'bottom');                
            end
            
            Pl.patch(x0 + it2t([t_st, t_en]), ...
                y0 + [-1, 0] * bar_height - bar_margin, C{:});
        end

        hold off;
        bml.plot.beautify;
        set(get(gca, 'XAxis'), 'visible', 'off');
        set(get(gca, 'YAxis'), 'visible', 'off');

        % set(gca, 'YTick', []);
        xlim([-0.1, t(end)]);
%         ylim([-2 * bar_height - bar_margin, max(thres) + 0.1]);
    end
    function plot_targetwise(Pl, slope, color)
        %%
        xy0 = Pl.xy0;
        x0 = Pl.xy0(1);
        y0 = Pl.xy0(2);
        
        dt = 0.01;
        t = 0:dt:2.3;
        x_lim = [-0.2, t(end)];
        slope0 = Pl.slope0;
        thres0 = Pl.thres0;
        xlim(x_lim);

        thres = 1;
        bar_margin = 0.2;
        bar_height = 0.2;

        line_width = 2;

        thres_sum = thres0 + thres;
        thres_fac = thres0 / thres_sum;
        
        slope0 = slope0 * thres_fac;
        slope = slope * thres_fac;
        thres0 = thres0 * thres_fac;
        thres = thres * thres_fac;
        
        slope01 = slope0 + slope;
        thres01 = thres0 + thres;
        
        t01 = thres01 / slope01;
%         t01 = t(find(slope01 * t >= thres01, 1, 'first'));
        
        plot(x0 + t, y0 + min(slope01 * t, thres01), '-', ...
            'Color', 'k', 'LineWidth', line_width);
        hold on;
        plot(x0 + t, y0 + min(slope01 * t, thres01), '--', ...
            'Color', color, 'LineWidth', line_width);

        height0 = slope0 * t01;
        plot(x0 + t, y0 + min(slope0 * t, slope0 * t01), '-', ...
            'Color', 'k', 'LineWidth', line_width);
        hold on;
        plot(x0 + t, y0 + min(slope * t, slope * t01), '-', ...
            'Color', color, 'LineWidth', line_width);
        
        color0 = 0.75 + zeros(1,3);
        h = plot(x0 + [-0.05, t01], y0 + [0 0] + height0, ':', ...
            'Color', color0, 'LineWidth', line_width/2);
        uistack(h, 'bottom');
        ylim(Pl.y_lim);
        xlim(x_lim);
        arrow(xy0 + [-0.1, 0], xy0 + [-0.1, height0-0.05], Pl.arrow_style{:}, ...
            'Color', color0);
        
        text(x0 + t01, y0 + mean([max(slope0 * t01, slope * t01), thres01]), ...
            '| |', 'FontSize', 6);
        text(x0 + t01, y0 + mean([slope0 * t01, slope * t01]), ...
            '+');
        xlim(x_lim);
        
%         Pl.patch([0, t01], [-1, 0] * bar_height - bar_margin, ...
%             'k');
%         Pl.patch([0, t01], [-2, -1] * bar_height - bar_margin, ...
%             color);
%         Pl.patch([0, thres/slope], [-2, -1] * bar_height - bar_margin, color);
%         Pl.patch([0, thres0/slope0], ...
%             [-1, 0] * bar_height - bar_margin, 'k');

        hold off;
        bml.plot.beautify;
        set(get(gca, 'XAxis'), 'visible', 'off');
        set(get(gca, 'YAxis'), 'visible', 'off');

        % set(gca, 'YTick', []);
%         ylim([-2 * bar_height - bar_margin, thres01 + 0.1]);
    end
end
methods
    function [ys_easy, ys_hard, he, hh] = plot_rt_cond(Pl, fac, colors)
        %%
        axis equal
        
        line_width = 2;
        if nargin < 3
            colors = row2cell(bml.plot.colormaps.cool2(2));
        end            
%         color_easy = bml.plot.color_lines('b');
%         color_hard = bml.plot.color_lines('r');
        color_easy = colors{1};
        color_hard = colors{2};
        x = -2:0.01:2;
        
        ye = @(x) normpdf(x) / normpdf(0) + 0.3;
        yh = @(x) normpdf(x) / normpdf(0) * fac + 1;
%         xlim(x([1, end]) * 1.1);
%         ylim([0, max(yh(x)) * 1.1]);

        ax1 = axis;
        he = plot(x, ye(x), '-', 'Color', color_easy, 'LineWidth', line_width);
        hold on;
        hh = plot(x, yh(x), '-', 'Color', color_hard, 'LineWidth', line_width);
        
        xs = -2:0;
        ys_easy = ye(xs);
        ys_hard = yh(xs);
        
        x_shift = 0.03;
        axis(ax1);
        Pl.plot_rt_arrows(xs - x_shift, ys_easy, color_easy);
        Pl.plot_rt_arrows(xs + x_shift, ys_hard, color_hard);
        
        axis off;
    end
    function plot_rt_arrows(Pl, xs, ys1, color1)
        C = varargin2C({
            'Length', 1 % 2.5
            }, Pl.arrow_style);
        ax1 = axis;
        for ix = 1:numel(xs)
            x = xs(ix);
            y = ys1(ix) - 0.15;
            arrow([x, 0], [x, y], C{:}, 'Color', color1);
            axis(ax1);
        end
    end
    function plot_rt_arrows_horz(Pl, xs, ys1, color1)
        ax1 = axis;
        for ix = 1:numel(xs)
            x = xs(ix);
            y = ys1(ix) - 0.05;
            arrow([0, x], [y, x], Pl.arrow_style{:}, 'Color', color1);
            axis(ax1);
        end
    end
    function plot_rt_rt(Pl, ys_easy, ys_hard, varargin)
        S = varargin2S(varargin, {
            'colors', row2cell(bml.plot.colormaps.cool2(2));
            'colors_markers', []
            'LineStyle', '-'
            });
        %%
%         color_easy = bml.plot.color_lines('b');
%         color_hard = bml.plot.color_lines('r');
        color_easy = S.colors{1};
        color_hard = S.colors{2};
        
        n = length(ys_easy);
        if isempty(S.colors_markers)
            colors_markers = row2cell(bml.plot.colormaps.winter2(n));
        else
            colors_markers = S.colors_markers;
        end

        ax1 = axis;
%         axis equal;
        max_y = max(ys_hard) * 1.1;
        ylim([0, max_y]);
        
        hold on;
        max_y_easy = max(ys_easy) * 1.5;
        plot([0, max_y_easy], ...
             [0, max_y_easy], '--', 'Color', 0.7 + [0 0 0]);
        hold on;
        
        b = glmfit(ys_easy, ys_hard, 'normal');
        x = ys_easy([1, end]) + [-0.2, 0.2];
        y = glmval(b, x, 'identity');
        plot(x, y, 'k', ...
            'LineStyle', S.LineStyle, ...
            'LineWidth', 1);
        hold on;
        
        for ii = 1:n
            plot(ys_easy(ii), ys_hard(ii), 'o', ...
                'MarkerFaceColor', colors_markers{ii}, ...
                'MarkerEdgeColor', 'w');
            hold on;
        end
        
        Pl.plot_rt_arrows_horz((-1:1)*0.09, ys_easy, color_easy);
%         Pl.plot_rt_arrows_horz(zeros(1,3), ys_easy, color_easy);
%         Pl.plot_rt_arrows(ys_easy - 0.02, ys_easy, color_easy);
        Pl.plot_rt_arrows(ys_easy + 0.02, ys_hard - 0.05, color_hard);
        
        axis(ax1);
        axis off;
        
%         x_lim = xlim;
%         y_lim = ylim;
%         xlim([0, x_lim(2)]);
%         ylim([0, y_lim(2)]);
    end
    function h = plot_ch_cond(Pl, slope, color)
        x = -2:0.01:2;
        y = invLogit(x * slope);
        h = plot(x, y, '-', 'Color', color, 'LineWidth', 2);
    end
end
%% Utilities
methods
    function patch(~, x, y, c, varargin)
        C = varargin2C(varargin, {
            'EdgeColor', 'none'
            });
            
        patch(x([1,2,2,1]), y([1,1,2,2]), ...
            c, C{:});
    end
end
end