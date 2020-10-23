classdef Main < Fit.D2.Bounded.Main
    % Fit.D2.Targetwise.Main
    %
    % 2015 YK wrote the initial version.

%% Main
methods
    function W = Main(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
end
%% Object properties
methods
    function set_Dtb(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Dtb = ...
            W.enforce_class('Fit.D2.Targetwise.Dtb', ...
                obj_or_name);
        W.set_sub_from_props({'Dtb'});
    end
end
end