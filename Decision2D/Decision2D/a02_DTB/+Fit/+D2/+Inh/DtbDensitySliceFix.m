classdef DtbDensitySliceFix < Fit.D2.Inh.DtbDensitySliceFree
properties
    slprops0 = [0.5, 0.5];
end
methods
    function init(W, varargin)
        W.init@Fit.D2.Inh.DtbDensitySliceFree(varargin{:});
        
        lb_logit = logit(0.04);
        ub_logit = -lb_logit;
        
        logit_slprops0 = logit(W.slprops0);
        logit_slprops0 = max(min(logit_slprops0, ub_logit), lb_logit);
        
        W.th0.logit_slprop1 = logit_slprops0(1);
        W.th0.logit_slprop2 = logit_slprops0(2);
        W.fix_to_th0_('logit_slprop1');
        W.fix_to_th0_('logit_slprop2');
    end 
    function fs = get_file_fields0(W)
        fs = [
            W.get_file_fields0@Fit.D2.Inh.DtbDensitySliceFree
            {
            'slprops0', 'sl0'
            }
            ];
    end
end
end