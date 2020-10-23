classdef DriftConst < Fit.D2.Common.Drift
    % Fit.D2.Common.DriftConst
    %
    % 2015 YK wrote the initial version.

methods
    function W = DriftConst(varargin)
        W = W@Fit.D2.Common.Drift(varargin{:});
    end
end
end