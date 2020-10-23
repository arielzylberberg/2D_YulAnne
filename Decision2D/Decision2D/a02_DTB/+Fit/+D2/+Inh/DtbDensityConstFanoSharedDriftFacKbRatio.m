classdef DtbDensityConstFanoSharedDriftFacKbRatio < ...
        Fit.D2.Inh.DtbDensityConstFanoSharedDriftFac ...
        & Fit.D2.RT.Bounded.KBRatioPairDrift
    %
    % 2015 YK wrote the initial version.
methods
    function add_params0(W)
        W.add_params0@Fit.D2.Inh.DtbDensityConstFanoSharedDriftFac;
        W.replace_kb_w_ratio;
    end
    function pred(W)
        W.get_kb_from_ratio;
        W.pred@Fit.D2.Inh.DtbDensityConstFanoSharedDriftFac;
    end
end
end