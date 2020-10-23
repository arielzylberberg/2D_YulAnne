classdef Plot < Fit.D2.Bounded.Plot
    % Fit.D2.Inh.Plot
    %
    % alias
methods
    function Plt = Plot(varargin)
        Plt = Plt@Fit.D2.Bounded.Plot(varargin{:});
    end
end
end