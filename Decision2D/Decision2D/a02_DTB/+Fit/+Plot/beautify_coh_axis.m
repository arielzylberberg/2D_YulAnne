function beautify_coh_axis(ax)
% beautify_coh_axis(ax, varargin)
%
% OPTIONS:
% 'feat', Fit.Plot.determine_feat(ax);

    if ~exist('ax', 'var'), ax = gca; end
    
    xy = bml.plot.get_all_xy(ax);
    if isempty(xy)
        return;
    end
    
    min_x = min(xy(:,1));
    max_x = max(xy(:,1));
    
    labels = Fit.Plot.beautify_coh_labels([min_x, nan, 0, nan, max_x]);
    
    set(ax, ...
        'XTick', linspace(min_x, max_x, 5), ...
        'XTickLabel', labels, ...
        'XGrid', 'off', ...
        'TickLength', [0.01, 0.05]);
%     set(ax, 'XTick', [-.512, -.256, 0, .256, .512]);
%     set(ax, 'XTickLabel', {'-51.2', '-25.6', '0', '25.6', '51.2'});

    xlim(ax, [min_x, max_x] * 1.1);
    
    Fit.Plot.xlabel(ax);
end