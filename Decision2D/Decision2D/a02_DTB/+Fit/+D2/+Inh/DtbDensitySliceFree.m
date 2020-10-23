classdef DtbDensitySliceFree < Fit.D2.Inh.DtbDensityIndivJt
    % With two parameters, control how each stream uses time.
    % dim1 gets slprop1, and dim2 gets slprop2, and each is in [0, 1].
    % drift and sigmaSq for each dim linearly scales 
    % with slprop1 and slprop2.
    %
    % 2017 YK wrote the initial version.
methods
    function W = DtbDensityIndivJt(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end    
    function init(W, varargin)
        W.init@Fit.D2.Inh.DtbDensityIndivJt(varargin{:});
        
        % Fix all params and add slprop
        W.th.p_dim1_1st = 1;
        W.fix_to_th_('p_dim1_1st');
        
        W.add_params({
            {'logit_slprop1', logit(0.5), logit(0.04), logit(0.96)}
            {'logit_slprop2', logit(0.5), logit(0.04), logit(0.96)}
            });
    end
    function pred(W)
        % Scale drift and sigmaSq_fac
        slprop1 = invLogit(W.th.logit_slprop1);
        slprop2 = invLogit(W.th.logit_slprop2);
        
        % Fix p_dim1_1st = 1 and modulate 1_1 and 2_1.
        % (The ones in effect when dim1 is 1st.)
        W.th.p_dim1_1st = 1;
        W.th.drift_fac_together_dim1_1 = slprop1;
        W.th.sigmaSq_fac_together_dim1_1 = slprop1;
        W.th.drift_sigmaSq_fac_dim1_1 = slprop1;
        
        W.th.drift_fac_together_dim2_1 = slprop2;
        W.th.sigmaSq_fac_together_dim2_1 = slprop2;
        W.th.drift_sigmaSq_fac_dim2_1 = slprop2;
        
        % Set to arbitrary number
        % : *_2 won't be used because p_dim1_1st = 1
        W.th.drift_sigmaSq_fac_dim1_2 = 1;
        W.th.drift_sigmaSq_fac_dim2_2 = 1;        
        W.th.drift_fac_together_dim1_2 = 1;
        W.th.drift_fac_together_dim2_2 = 1;
        W.th.sigmaSq_fac_together_dim1_2 = 1;
        W.th.sigmaSq_fac_together_dim2_2 = 1;
        
        W.fix_to_th_({
            'p_dim1_1st'
            'drift_fac_together_dim1_1'
            'drift_fac_together_dim2_1'
            'sigmaSq_fac_together_dim1_1'
            'sigmaSq_fac_together_dim2_1'
            'drift_fac_together_dim1_2'
            'drift_fac_together_dim2_2'
            'sigmaSq_fac_together_dim1_2'
            'sigmaSq_fac_together_dim2_2'
            'drift_sigmaSq_fac_dim1_1'
            'drift_sigmaSq_fac_dim1_2'
            'drift_sigmaSq_fac_dim2_1'
            'drift_sigmaSq_fac_dim2_2'
            });
        W.remove_constraints_by_params({
            'p_dim1_1st'
            'drift_fac_together_dim1_1'
            'drift_fac_together_dim2_1'
            'sigmaSq_fac_together_dim1_1'
            'sigmaSq_fac_together_dim2_1'
            'drift_fac_together_dim1_2'
            'drift_fac_together_dim2_2'
            'sigmaSq_fac_together_dim1_2'
            'sigmaSq_fac_together_dim2_2'
            'drift_sigmaSq_fac_dim1_1'
            'drift_sigmaSq_fac_dim1_2'
            'drift_sigmaSq_fac_dim2_1'
            'drift_sigmaSq_fac_dim2_2'
            });
        
        % Inherit the rest
        W.pred@Fit.D2.Inh.DtbDensityIndivJt;
    end
end
end