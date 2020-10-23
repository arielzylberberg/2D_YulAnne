classdef Dtb1D < Fit.D1.Bounded.Dtb & Fit.D2.Common.CommonWorkspace
    % Fit.D2.Bounded.Dtb1D
    %
    % 2015 YK wrote the initial version.
    
%% Initialization
methods
    function W = Dtb1D(varargin)
        % Dtb1D('dim_rel_W', 1 or 2, ...)
        
        W.set_Data;
        W.set_Drift;
        W.set_SigmaSq;
        
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function set_Data(W, obj_or_name)
        if nargin < 2, obj_or_name = Fit.D2.Common.DataChRtPdf; end
        obj_or_name = W.enforce_class('Fit.D2.Common.DataChRtPdf', obj_or_name);
        W.set_Data@Fit.Common.CommonWorkspace(obj_or_name);
    end    
    function set_Drift(W, obj_or_name)
        args = {'dim_rel_W', W.get_dim_rel_W};
        
        if nargin < 2, obj_or_name = 'Const'; end
        W.Drift = W.enforce_class('Fit.D2.Common.Drift', obj_or_name);
        varargin2props(W.Drift, ...
            args);
        W.set_sub_from_props({'Drift'});
        W.Drift.customize_th_for_Data(W.get_dim_rel_W);
    end
    function set_SigmaSq(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.SigmaSq = W.enforce_class('Fit.D2.Common.SigmaSq', obj_or_name);
        
        W.SigmaSq.set_dim_rel_W(W.get_dim_rel_W);
        
        W.set_sub_from_props({'SigmaSq'});
    end
end
end