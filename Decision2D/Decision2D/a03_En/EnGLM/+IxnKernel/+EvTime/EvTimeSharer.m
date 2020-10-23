classdef EvTimeSharer ...
        < IxnKernel.EvTime.EvAxis.EvidenceAxisInheritable ...
        & IxnKernel.EvTime.TimeAxis.TimeInheritable ...
        & FitWorkspace ...
        & bml_local.oop.PropFileNameTree
%% === EvTimeSharer ===
methods
    function Ev = EvTimeSharer(varargin)
        if nargin > 0
            Ev.init(varargin{:});
        end
    end
    function init(Ev, varargin)
        varargin2props(Ev, varargin, true);
    end
end
methods
    %% Always use root's Time and EvAxis.
    function set_Time(W, Time)
        root = W.get_Data_source;
        root.set_Time@IxnKernel.EvTime.TimeAxis.TimeInheritable(Time);
    end
    function Time = get_Time(W)
        root = W.get_Data_source;
        Time = root.get_Time@IxnKernel.EvTime.TimeAxis.TimeInheritable;
    end
    function set_EvAxis(W, EvAxis)
        root = W.get_Data_source;
        root.set_EvAxis@IxnKernel.EvTime.EvAxis.EvidenceAxisInheritable(EvAxis);
    end
    function EvAxis = get_EvAxis(W)
        root = W.get_Data_source;
        EvAxis = root.get_EvAxis@IxnKernel.EvTime.EvAxis.EvidenceAxisInheritable;
    end
    function set_root(W, new_root)
        % When the W itself becomes a root,
        % set its Time & EvAxis to the previous root's Time & EvAxis.
        
        prev_root = W.get_root;
        W.set_root@FitWorkspace(new_root);
        if W.is_root % Equivalent to W == new_root
            W.set_Time(prev_root.get_Data);
            W.set_EvAxis(prev_root.get_EvAxis);
        end
    end
    function src = get_Data_source(W)
        % Defaults to the root. 
        % Modify, e.g., to self, in subclasses if necessary.
        % Then set_root should be changed as well.
        src = W.get_root;
    end
end
end