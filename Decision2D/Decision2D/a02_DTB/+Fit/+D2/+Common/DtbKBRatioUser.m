classdef DtbKBRatioUser < Fit.D2.Bounded.Dtb1D
properties
    kb_ratio = true;
    KBRatio = [];
end
methods
    function set_KBRatio(W, KB)
        % set_KBRatio(W, KB)
        if ~exist('KB', 'var')
            KB = Fit.D2.Common.KBRatio;
        end
        assert(isa(KB, 'Fit.D2.Common.KBRatio'));

        W.KBRatio = KB;
        KB.set_dim_rel_W(W.get_dim_rel_W);
        KB.set_Drift(W.Drift);
        KB.set_Bound(W.Bound);
        
        W.set_sub_from_props('KBRatio');
        if W.Data.loaded
            W.customize_th_for_Data;
        end
    end
    function set_Drift(W, varargin)
        if ~isempty(W.KBRatio)
            W.KBRatio.set_Drift(W.Drift);
        end
    end
    function set_Bound(W, varargin)
        if ~isempty(W.KBRatio)
            W.KBRatio.set_Bound(W.Bound);
        end
    end
    function set_dim_rel_W(W, varargin)
        W.set_dim_rel_W@Fit.D2.Common.CommonWorkspace(varargin{:});
        W.KBRatio.set_dim_rel_W(W.get_dim_rel_W);        
    end
    function customize_th_for_Data(W)
        if W.kb_ratio
            W.customize_KBRatio_for_Data;
        end
    end
    function customize_KBRatio_for_Data(W)
        if ~isempty(W.KBRatio)
            W.KBRatio.customize_th_for_Data;
        end
    end
    function calc_KBRatio(W)
        if ~isempty(W.KBRatio)
            W.KBRatio.pred;
        end
    end
end    
end