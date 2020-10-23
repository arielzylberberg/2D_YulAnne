classdef Dtb < Fit.D2.Inh.Dtb
    % Fit.D2.RT.BoundedCondEn.Dtb
    %
    % Trialwise calculation
    %
    % 2015 YK wrote the initial version.
properties (SetAccess = protected)
    Drift % Unlike regular drifts, gives a drift for each trial.
    Bound
    TndSt % Duration of the stimulus ignored at the beginning
end
methods
    function W = Dtb
        W.add_deep_copy({'Drift', 'Bound', 'TndSt'});
        W.set_Drift;
        W.set_Bound;
        W.set_TndSt;
    end
    function pred(W)
        W.Data.set_Td_pred_pdf_tr(W.get_Td_pdf_tr);
    end
end
%% Main calculation
methods
    function [Td_pdf_tr, trajs] = get_Td_pdf_tr(W)
        % [Td_pdf_tr, trajs] = get_Td_pdf(W)
        % trajs is {traj_dim1_1st, traj_dim1_2nd}, where each element
        % is traj from the first repetition of each batch.
        error('Modify in subclasses!');
    end        
end
%% Dtb parameters
methods
    function TndSt_pdf = get_TndSt_pdf(W)
        % pdf_tnd_st: nt x 2 matrix
        TndSt_pdf = W.TndSt.get_pdf_tnd;
        assert(isequal(size(TndSt_pdf), W.get_TndSt_pdf_size));
    end 
    function size_ = get_TndSt_pdf_size(W)
        size_ = [W.Time.get_nt, 2];
    end
end
%% Get/Set - objects for trial-by-trial drift.
methods
    function set_Data(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        obj_or_name = ...
            W.enforce_class('Fit.D2.Common.DataChRtPdfEn', ...
                obj_or_name);
        W.set_Data@Fit.D2.Common.CommonWorkspace(obj_or_name);
    end
    function set_Drift(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Drift = W.enforce_class('Fit.D2.RT.BoundedCondEn.Drift', obj_or_name);
        W.set_sub_from_props({'Drift'});
    end
    function set_Bound(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Bound = W.enforce_class('Fit.D2.Common.Bound', obj_or_name);
        W.set_sub_from_props({'Bound'});
    end
    function set_TndSt(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.TndSt = W.enforce_class('Fit.D1.Bounded.Tnd', obj_or_name);
        W.set_sub_from_props({'TndSt'});
    end
end
%% Demo
methods (Static)
    function Dtb = demo
        %%
        Dtb = my_class; % Fit.D2.RT.BoundedCondEn.Dtb;
        Dtb.Data.load_data;
        
        %%
        Dtb.pred;
        
    end
end
end