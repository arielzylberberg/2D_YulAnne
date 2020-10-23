classdef Dtb < Fit.D2.Bounded.DtbIndepDim
    % Fit.D2.IrrIxn.Dtb
    %
    % 2016 YK wrote the initial version.
    
%% Initialization
methods
    function W = Dtb(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init_child(W, child_name, varargin)
        switch child_name
            case {'Dtb1', 'Dtb2'}
                dim_rel_W = str2double(child_name(end));
                C = varargin2C({
                    'dim_rel_W', dim_rel_W
                    }, varargin);
                W.children.(child_name).init(C{:});
            otherwise
                W.children.(child_name).init(varargin{:});
        end
    end    
    function set_Dtb1(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Dtb1 = W.enforce_class('Fit.D2.IrrIxn.Dtb1D', obj_or_name);
        varargin2props(W.Dtb1, ...
            {'dim_rel_W', 1});
        W.set_sub_from_props({'Dtb1'});
        
        % Drift biases are independently estimated from irr_cond
        % without direct reference to the parameters of Drift_irr
    end
    function set_Dtb2(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Dtb2 = W.enforce_class('Fit.D2.IrrIxn.Dtb1D', obj_or_name);
        varargin2props(W.Dtb2, ...
            {'dim_rel_W', 2});
        W.set_sub_from_props({'Dtb2'});
        
        % Drift biases are independently estimated from irr_cond
        % without direct reference to the parameters of Drift_irr
    end
    function set_Drift(W, name)
        assert(ischar(name));
        W.Dtb1.set_Drift(name);
        W.Dtb2.set_Drift(name);
    end
    function set_Bound(W, name)
        assert(ischar(name));
        W.Dtb1.set_Bound(name);
        W.Dtb2.set_Bound(name);
    end
    function set_SigmaSq(W, name)
        assert(ischar(name));
        W.Dtb1.set_SigmaSq(name);
        W.Dtb2.set_SigmaSq(name);
    end
    function set_Td(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Ser'; end
        W.Td = W.enforce_class('Fit.D2.IrrIxn.Td', obj_or_name);
        W.set_sub_from_props({'Td'});
    end
    function customize_th_for_Data(W)
        for dim = 1:2
            W.Dtbs{dim}.customize_th_for_Data;
        end
    end
end
end