classdef DtbWithDtbCalc < Fit.D2.RT.BoundedCondEn.Dtb
    %
    % 2015 YK wrote the initial version.
properties (SetAccess = private)
    DtbCalc
end
methods
    function W = DtbWithDtbCalc
        W.add_deep_copy({'DtbCalc'});        
        W.set_DtbCalc;
    end
    function [Td_pdf_tr, trajs] = get_Td_pdf_tr(W)
        % [Td_pdf_tr, trajs] = get_Td_pdf(W)
        % trajs is {traj_dim1_1st, traj_dim1_2nd}, where each element
        % is traj from the first repetition of each batch.
        
        %% Simulate - may concatenate parameters to parallelize
        W.DtbCalc.remove_DtbCalc_all;
        for dim_1st = 1:2
            C = varargin2C({
                'n_tr', W.Data.get_n_tr
                'drift', W.get_drift_cond_t
                'bound', W.get_bound_t_ch
                'tnd_st', W.get_TndSt_pdf
                'sigmaSq_fac_bef_start', hVec(W.sigmaSq_fac_bef_start)
                'sigmaSq_fac_together',  hVec(W.sigmaSq_fac_together(:, dim_1st))
                'drift_fac_together',  hVec(W.drift_fac_together(:, dim_1st))
                });
            W.DtbCalc.add_DtbCalc(C, {
                'dim_1st', dim_1st
                });
        end       
        
        [Td_pdfs, trajs] = W.DtbCalc.get_pred_td_tr_t_ch_pdf;
        p_dim1_1st = W.th.p_dim1_1st;
        
        Td_pdf_tr = Td_pdfs{1} * p_dim1_1st + Td_pdfs{2} * (1 - p_dim1_1st);
    end
    function set_DtbCalc(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.DtbCalc = W.enforce_class('Fit.D2.RT.BoundedCondEn.DtbCalcCollectionSim', obj_or_name);
        W.set_sub_from_props({'DtbCalc'});
    end
end
end