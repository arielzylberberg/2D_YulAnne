classdef DtbIndepDim < Fit.D2.Bounded.Dtb
    % Fit.D2.Bounded.DtbIndepDim
    %
    % 2015 YK wrote the initial version.
properties (Dependent)
    td_kind
end
properties (SetAccess = protected)
    Dtb1
    Dtb2
    Td    
end
properties (Dependent)
    Dtbs
end
methods
    function W = DtbIndepDim(varargin)
        W.add_deep_copy({'Dtb1', 'Dtb2', 'Td'});
        
        W.set_Data;
        W.set_Dtb1;
        W.set_Dtb2;
        W.set_Td;
        
        bml.oop.varargin2props(W, varargin, true);
    end
    function pred(W)
        W.Data.set_Td_pred_pdf( ...
            W.Td.get_Td_pdf( ...
                {W.Dtb1.get_Td_pdf, W.Dtb2.get_Td_pdf}));
    end
    %% Get/Set
    function Dtbs = get.Dtbs(W)
        Dtbs = {W.get_Dtb1, W.get_Dtb2};
    end
    function set.Dtbs(W, v)
        assert(iscell(v) && numel(v) == 2);
        W.set_Dtb1(v{1});
        W.set_Dtb2(v{2});
    end
    function set_Dtb1(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Dtb1 = W.enforce_class('Fit.D2.Bounded.Dtb1D', obj_or_name);
        varargin2props(W.Dtb1, ...
            {'dim_rel_W', 1});
        W.set_sub_from_props({'Dtb1'});
    end
    function set_Dtb2(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Dtb2 = W.enforce_class('Fit.D2.Bounded.Dtb1D', obj_or_name);
        varargin2props(W.Dtb2, ...
            {'dim_rel_W', 2});
        W.set_sub_from_props({'Dtb2'});
    end
    function set_Td(W, obj_or_name)
        if nargin < 2, obj_or_name = W.td_kind; end
        W.Td = W.enforce_class('Fit.D2.Bounded.Td', obj_or_name);
        W.set_sub_from_props({'Td'});
    end
    function v = get_Dtb1(W)
        v = W.Dtb1;
    end
    function v = get_Dtb2(W)
        v = W.Dtb2;
    end
    function set.td_kind(W, v)
        W.set_Td(v);
    end
    function v = get.td_kind(W)
        if isempty(W.Td)
            v = 'Ser';
        else
            v = strrep(bml.pkg.pkg2class(class(W.Td)), 'Td', '');
        end
    end
end
%% Delegation
properties (Dependent)
    Drifts
    Drift1
    Drift2
    Bounds
    Bound1
    Bound2
    SigmaSqs
    SigmaSq1
    SigmaSq2
end
methods
    function v = get.Drifts(W)
        v = {W.Drift1, W.Drift2};
    end
    function set.Drifts(W, v)
        W.Drift1 = v{1};
        W.Drift2 = v{2};
    end

    function v = get.Drift1(W)
        v = W.Dtb1.Drift;
    end
    function set.Drift1(W, v)
        W.Dtb1.Drift = v;
    end

    function v = get.Drift2(W)
        v = W.Dtb2.Drift;
    end
    function set.Drift2(W, v)
        W.Dtb2.Drift = v;
    end

    function v = get.Bounds(W)
        v = {W.Bound1, W.Bound2};
    end
    function set.Bounds(W, v)
        W.Bound1 = v{1};
        W.Bound2 = v{2};
    end

    function v = get.Bound1(W)
        v = W.Dtb1.Bound;
    end
    function set.Bound1(W, v)
        W.Dtb1.Bound = v;
    end

    function v = get.Bound2(W)
        v = W.Dtb2.Bound;
    end
    function set.Bound2(W, v)
        W.Dtb2.Bound = v;
    end

    function v = get.SigmaSqs(W)
        v = {W.SigmaSq1, W.SigmaSq2};
    end
    function set.SigmaSqs(W, v)
        W.SigmaSq1 = v{1};
        W.SigmaSq2 = v{2};
    end

    function v = get.SigmaSq1(W)
        v = W.Dtb1.SigmaSq;
    end
    function set.SigmaSq1(W, v)
        W.Dtb1.SigmaSq = v;
    end

    function v = get.SigmaSq2(W)
        v = W.Dtb2.SigmaSq;
    end
    function set.SigmaSq2(W, v)
        W.Dtb2.SigmaSq = v;
    end    
end
end