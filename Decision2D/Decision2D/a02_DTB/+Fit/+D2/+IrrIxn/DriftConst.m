classdef DriftConst < Fit.D2.IrrIxn.Drift
    % Fit.D2.IrrIxn.DriftConst
    %
    % 2015 YK wrote the initial version.
methods
    function W = DriftConst(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
end
end