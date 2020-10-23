function beautify_ch_axis(ax, varargin)
    if ~exist('ax', 'var'), ax = gca; end
    
    S = varargin2S(varargin, {
        'feat', Fit.Plot.determine_feat(ax);
        });

    set(ax, ...
        'YTick', 0:0.5:1, ...
        'TickLength', [0.01, 0.05], ...
        'YTickLabel', {'0', ' ', '1'}, ...
        'YGrid',' off');
    
    switch S.feat
        case 'M'
            ylabel(ax, 'P_{right}');
        case 'C'
            ylabel(ax, 'P_{blue}');
    end
    ylim(ax, [-0.05, 1]);
end