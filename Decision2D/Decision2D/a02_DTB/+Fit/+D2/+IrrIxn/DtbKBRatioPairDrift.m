classdef DtbKBRatioPairDrift < Fit.D2.Bounded.DtbKBRatioPairDrift
    % Fit.D2.IrrIxn.DtbKBRatioPairDrift
methods
    function W = DtbKBRatioPairDrift(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function set_Dtb(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Dtb = W.enforce_class('Fit.D2.IrrIxn.Dtb1D', obj_or_name);
        W.set_sub_from_props({'Dtb'});
    end
end
end