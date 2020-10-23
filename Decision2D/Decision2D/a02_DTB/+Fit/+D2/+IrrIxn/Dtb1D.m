classdef Dtb1D < Fit.D2.Bounded.Dtb1D ...
        & Fit.D2.Common.DtbKBRatioUser
    % Fit.D2.IrrIxn.Dtb1D
    %
    % 2016 YK wrote the initial version.
methods
    function W = Dtb1D(varargin)
        W.kb_ratio = false;
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        W.init@Fit.D2.Bounded.Dtb1D(varargin{:});
        
        if W.kb_ratio
            W.set_KBRatio;        
        end
    end
    function set_dim_rel_W(W, varargin)
        W.set_dim_rel_W@Fit.D2.Bounded.Dtb1D(varargin{:});
        if ~isempty(W.Drift)
            W.Drift.set_dim_rel_W(varargin{:});
        end
        if ~isempty(W.SigmaSq)
            W.SigmaSq.set_dim_rel_W(varargin{:});
        end
    end
    function set_Drift(W, obj_or_name)
        args = {'dim_rel_W', W.get_dim_rel_W};
        
        if nargin < 2, obj_or_name = Fit.D2.IrrIxn.DriftConst; end
        W.Drift = W.enforce_class('Fit.D2.IrrIxn.Drift', ...
            obj_or_name);
        varargin2props(W.Drift, ...
            args);
        W.Drift.set_dim_rel_W(W.get_dim_rel_W);
        W.set_sub_from_props({'Drift'});
        W.set_Drift@Fit.D2.Common.DtbKBRatioUser(W.Drift);
        
        % SigmaSq for each joint condition is calculated separately
        % based on Drift.
        if ~isempty(W.SigmaSq)
            W.SigmaSq.set_Drift(W.Drift);
        end
    end
    function set_Bound(W, varargin)
        W.set_Bound@Fit.D2.Bounded.Dtb1D(varargin{:});
        W.set_Bound@Fit.D2.Common.DtbKBRatioUser(W.Bound);
    end
    function set_SigmaSq(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.SigmaSq = W.enforce_class('Fit.D2.IrrIxn.SigmaSq', obj_or_name);
        
        W.SigmaSq.set_dim_rel_W(W.get_dim_rel_W);        
        W.set_sub_from_props({'SigmaSq'});
        
        % Drift biases are independently estimated from irr_cond
        % without direct reference to the parameters of Drift_irr
        
        % SigmaSq for each joint condition is calculated separately
        % based on Drift.
        W.SigmaSq.set_Drift(W.Drift);
    end    
end
%% Import results from logistic fit
methods
    function customize_KBRatio_for_Data(W)
        W.import_logistic_fit;
    end
    function import_logistic_fit(W)
        Lgt = W.get_logistic_fit;
        mdl = Lgt.mdl;
        
        Drift = W.Drift;
        KBRatio = W.KBRatio;
        
        % Get fit
        slope = table2array(mdl.Coefficients('r', 'Estimate'));
        slope_se = table2array(mdl.Coefficients('r', 'SE'));
        
        bias = table2array(mdl.Coefficients('(Intercept)', 'Estimate'));
        bias_se = table2array(mdl.Coefficients('(Intercept)', 'SE'));
        
        KBRatio.set_k_b_prod(slope / 2);
        
        Drift.th0.bias = -bias / slope;
        Drift.fix_to_th0_('bias');
        
        if ismember({'i'}, mdl.CoefficientNames)
            bias_irr = table2array(mdl.Coefficients('i', 'Estimate'));
            bias_irr_se = table2array(mdl.Coefficients('i', 'SE'));
        
            % Divide with slope so that it can be treated as
            % biasing the condition itself.
            Drift.th0.k_irr = bias_irr / slope;
        else
            Drift.th0.k_irr = 0;
            bias_irr_se = 0;
        end
%         Drift.fix_to_th0_('k_irr');
        
        if ismember({'ai'}, mdl.CoefficientNames)
            bias_abs_irr = table2array(mdl.Coefficients('ai', 'Estimate'));
            bias_abs_irr_se = table2array(mdl.Coefficients('ai', 'SE'));
            
            % Divide with slope so that it can be treated as
            % biasing the condition itself.
            Drift.th0.k_abs_irr = bias_abs_irr / slope;
        else
            Drift.th0.k_abs_irr = 0;
            bias_abs_irr_se = 0;
        end
%         Drift.fix_to_th0_('k_abs_irr');
    end
    function Lgt = get_logistic_fit(W)
        L = load(W.get_file_logistic_fit);
        res = bml.ds.find(L.ds, {
            'Subj', W.subj(1)
            'Parad', W.parad
            'N_Dim', W.n_dim_task
            'Feature', Data.Consts.dimNames_long{W.dim_rel_W}
            });
        L2 = load(res.file{1});
        Lgt = L2.W0;
    end
    function file = get_file_logistic_fit(W)
        switch W.parad
            case 'RT'
                file = 'Data_2D/Fit.Ch.Main/subj={DX,MA,VL}+parad={RT}+dim_rel_W={1,2}+n_dim_task={1,2}+n_sim=100.mat';
            case 'sh'
                file = 'Data_2D/Fit.Ch.Main/subj={DX,MA,VL,YK}+parad={sh}+dim_rel_W={1,2}+n_dim_task={1,2}+n_sim=100.mat';
        end
    end
end
%% Prediction
methods
    function Td_pdf = get_Td_pdf(W)
        Td_pdf = W.get_Td_pdf@Fit.D2.Bounded.Dtb1D;
        
        n_conds = W.Data.get_nConds;
        Td_pdf = reshape(Td_pdf, [W.get_nt, n_conds, 2]);
        
        if W.get_dim_rel_W == 2
            Td_pdf = permute(Td_pdf, [1 2 3 5 4]);
        end
    end
end
end