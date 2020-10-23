classdef SigmaSqQuad < Fit.D1.Bounded.SigmaSqQuad & Fit.D2.Common.SigmaSq
    % Fit.D2.Common.SigmaSqQuad
    %
    % 2016 YK wrote the initial version.
    
methods
    function W = SigmaSqQuad(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
end
end