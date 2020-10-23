classdef EnCosBases < FitWorkspace
    % Fit.D2.Common.EnCosBases
    %
    % 2015 YK wrote the initial version
properties
    Bases
end
methods
    function W = EnCosBases
        W.set_Data;
    end
    function set_Data(W, varargin)
        W.set_Data@FitWorkspace(varargin{:});
        W.adapt_Data;
    end
    function adapt_Data(W)
        warning('Modify adapt_Data in subclasses!');
    end
end
end