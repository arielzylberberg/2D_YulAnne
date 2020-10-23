classdef CommonWorkspaceD2 ...
        < IxnKernel.EvTime.CommonWorkspace
properties (Dependent)
    % props_to_share_Ev
    % : Properties that share Ev (EvTimeD1) as a property.
    %   Whenever W.Ev is set, their Ev is set to W.Ev.Evs{dim},
    %   i.e., prop{dim}.Ev = Ev.Evs{dim}
    props_to_share_Ev
end
properties
    props_to_share_Ev_ = {};
    n_dim = 2;
end
properties (Abstract)
    ev % (tr, t, dim)
    ch % (tr, dim)
    n_trial
end
methods
    function W = CommonWorkspaceD2(varargin)
        W.Ev = IxnKernel.EvTime.EvTimeD2;
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function set_Ev(W, Ev)
        W.set_Ev@IxnKernel.EvTime.CommonWorkspace(Ev);
        
        for prop = W.props_to_share_Ev(:)'
            for dim = 1:numel(W.Ev.Evs)
                W.(prop{1}){dim}.Ev = W.Ev.Evs{dim};
            end
        end
    end
    
    function set.props_to_share_Ev(W, v)        
        W.props_to_share_Ev_ = v;
        W.Ev = W.Ev; % invoke set_Ev
    end
    function v = get.props_to_share_Ev(Ev)
        v = Ev.props_to_share_Ev_;
    end
    
end
end