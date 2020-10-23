classdef DtbSigmaSq < Fit.D2.Inh.DtbDensity
properties (SetAccess = protected)
    SigmaSq1
    SigmaSq2
end
properties (Dependent)
    SigmaSqs
end
%% SigmaSq
methods
    function W = DtbSigmaSq(varargin)
        W.set_SigmaSqs;
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function v = get_SigmaSq(W, dim)
        v = W.SigmaSqs{dim};
    end
    
    function set.SigmaSqs(W, v)
        W.set_SigmaSqs(v);
    end
    function v = get.SigmaSqs(W)
        v = W.get_SigmaSqs;
    end
    
    function v = get_SigmaSqs(W)
        v = {W.SigmaSq1, W.SigmaSq2};
    end
    function set_SigmaSqs(W, v)
        if exist('v', 'var')
            if iscell(v)
                if isscalar(v)
                    v = rep_deep_copy(v, [1, 2]);
                end
                assert(numel(v) == 2);
                W.set_SigmaSq1(v{1});
                W.set_SigmaSq2(v{2});
            elseif ischar(v)
                W.set_SigmaSq1(v);
                W.set_SigmaSq2(v);
            end
        else
            W.set_SigmaSq1;
            W.set_SigmaSq2;
        end
    end
    function set_SigmaSq1(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.SigmaSq1 = W.enforce_class('Fit.D2.Common.SigmaSq', obj_or_name, {
            'dim_rel_W', 1
            });
        W.set_sub_from_props({'SigmaSq1'});
        W.SigmaSq1.customize_th_for_Data(1);
    end
    function set_SigmaSq2(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.SigmaSq2 = W.enforce_class('Fit.D2.Common.SigmaSq', obj_or_name, {
            'dim_rel_W', 2
            });
        W.set_sub_from_props({'SigmaSq2'});
        W.SigmaSq2.customize_th_for_Data(2);
    end
end
end