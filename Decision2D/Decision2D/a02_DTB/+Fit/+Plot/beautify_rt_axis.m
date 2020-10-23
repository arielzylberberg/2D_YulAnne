function beautify_rt_axis(ax, varargin)
if ~exist('ax', 'var'), ax = gca; end

bml.plot.beautify_lim('ax', ax, 'xy', 'y', varargin{:});
bml.plot.beautify_tick(ax, 'y', varargin{:});

ylabel('RT (s)');
set(ax, ...
    'YGrid', 'off', ...
    'TickLength', [0.01, 0.05]);
end