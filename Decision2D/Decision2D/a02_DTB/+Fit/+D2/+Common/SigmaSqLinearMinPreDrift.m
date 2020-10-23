classdef SigmaSqLinearMinPreDrift ...
         < Fit.D1.Bounded.SigmaSqLinearMinPreDrift ...
         & Fit.D2.Common.SigmaSq
    % Fit.D2.Common.SigmaSqLinearMinPreDrift
    %
    % 2016 YK wrote the initial version.
    
methods
    function W = SigmaSqLinearMinPreDrift(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
end
end