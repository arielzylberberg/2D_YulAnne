classdef Main < Fit.D2.Bounded.Main & Fit.D2.Common.CommonWorkspace
    % Fit.D2.IrrIxn.Main
    
    % 2016 YK wrote the initial version.
    
%% Initialization
methods
    function W = Main(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function set_Dtb(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        if ischar(obj_or_name)
            W.Dtb = Fit.D2.IrrIxn.(['Dtb' obj_or_name]);
        else
            W.Dtb = obj_or_name;
        end
        W.set_sub_from_props({'Dtb'});
    end
    function customize_th_for_Data(W)
        W.Dtb.customize_th_for_Data;
    end
end
end