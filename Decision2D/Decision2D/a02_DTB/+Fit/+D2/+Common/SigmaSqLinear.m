classdef SigmaSqLinear < Fit.D1.Bounded.SigmaSqLinearPreDrift & Fit.D2.Common.SigmaSq
    % Fit.D2.Common.SigmaSqLinear
    %
    % 2016 YK wrote the initial version.
    
methods
    function W = SigmaSqLinear(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
end
end