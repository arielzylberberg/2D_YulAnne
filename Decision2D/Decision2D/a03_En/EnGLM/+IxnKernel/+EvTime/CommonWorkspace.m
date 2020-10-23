classdef CommonWorkspace ...
        < IxnKernel.EvTime.EvTimeSharer
properties (Dependent)
    Ev
end
properties
    Ev_ = IxnKernel.EvTime.EvTimeD1;
end
methods
    function set.Ev(W, Ev)
        W.set_Ev(Ev);
    end
    function set_Ev(W, Ev)
        W.Ev_ = Ev;
        W.add_children_props('Ev');
    end
    
    function Ev = get.Ev(W)
        Ev = W.get_Ev;
    end
    function Ev = get_Ev(W)
        Ev = W.Ev_;
    end
end
end
