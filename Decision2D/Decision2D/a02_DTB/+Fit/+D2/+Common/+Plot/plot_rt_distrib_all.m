function ax_all1 = plot_rt_distrib_all(p, varargin)
% ax = p(t, cond1, cond2, ch1, ch2)

S = varargin2S(varargin, {
    'plot_args', {}
    't', []
    'dt', 1/75
    'hold', 'off'
    'y', 'smooth'
    'desc', ''
    'ylim', [0 1]
    });

n_cond1 = size(p, 2);
n_cond2 = size(p, 3);

nt = size(p, 1);
if isempty(S.t)
    S.t = (0:(nt-1))' * S.dt;
end

for ch1 = 2:-1:1
    for ch2 = 2:-1:1
        fig_tag(sprintf('%s_ch%d_%d_%s', S.desc, ch1, ch2, S.y));
        
        for cond1 = n_cond1:-1:1
            for cond2 = n_cond2:-1:1
                ax = subplotRC(n_cond1, n_cond2, cond2, cond1);
                
                p1 = p(:, cond1, cond2, ch1, ch2);
                
                switch S.y
                    case 'smooth'
                        y = smooth(p1);
                        plot(S.t, y, S.plot_args{:});
                    case 'cumsum'
                        y = cumsum(p1);
                        stairs(S.t(:), y(:), S.plot_args{:});
                end
                
                x_tick = 0:1:5;
                y_tick = -5:1:5;
                n_x_tick = length(x_tick);
                n_y_tick = length(y_tick);
                
                ax_all(cond1, cond2) = ax;
                
                bml.plot.crossLine('h', 0);
                
                hold(ax, S.hold);
            end
        end
        set(ax_all, 'XTick', x_tick, 'XGrid', 'on');
        set(ax_all, 'XTickLabel', repmat({''}, [1, n_x_tick]));
        set(ax_all, 'YTick', y_tick, 'YGrid', 'on');
        set(ax_all, 'YTickLabel', repmat({''}, [1, n_y_tick]));
        bml.plot.beautify(ax_all);
        
        set(ax_all, 'XLim', [0, S.t(end)]);
        switch S.y
            case 'smooth'
                sameAxes(ax_all);
                
            case 'cumsum'
                set(ax_all, 'YLim', S.ylim);
        end
                
        ax_all1{ch1,ch2} = ax_all;
    end
end
end