function xlabel(ax, varargin)
% xlabel(ax, varargin)
%
% OPTIONS:
% 'feat', Fit.Plot.determine_feat(ax);
    if ~exist('ax', 'var'), ax = gca; end
    
    S = varargin2S(varargin, {
        'feat', Fit.Plot.determine_feat(ax);
        });
    
    switch S.feat
        case 'M'
            xlabel(ax, 'Motion Strength (%)');
        case 'C'
            xlabel(ax, 'Color Strength (log odds blue)');
    end
end