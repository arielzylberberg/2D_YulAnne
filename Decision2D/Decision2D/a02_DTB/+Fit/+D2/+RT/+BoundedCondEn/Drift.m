classdef Drift < Fit.D2.Common.Drift
    % Fit.D2.RT.BoundedCondEn.Drift
    %
    % 2015 YK wrote the initial version.
properties (SetAccess = protected)
    Drift1
    Drift2
end
properties (Dependent)
    Drifts
end
methods
    function W = Drift
        W.add_deep_copy({'Drift1', 'Drift2'});
        
        W.remove_params_all; % Remove inherited parameters
        W.set_Drift1;
        W.set_Drift2;
    end
    %% Predictions
    function drift_cond_t = get_drift_cond_t(W)
        % drift_cond_t = get_drift_cond_t(W)
        %
        % drift_cond_t : n_trial x nt x ndim
        % Different from Fit.D1.Bounded.Drift
        
        drift_cond_t(:,:,1) = W.Drift1.get_drift_cond_t; % n_trial x nt
        drift_cond_t(:,:,2) = W.Drift2.get_drift_cond_t; % n_trial x nt
    end
    %% Get/Set objects
    function set_Data(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        obj_or_name = ...
            W.enforce_class('Fit.D2.Common.DataChRtPdfEn', ...
                obj_or_name);
        W.set_Data@Fit.D2.Common.Drift(obj_or_name);
    end
    function set.Drifts(W, v)
        W.set_Drifts(v);
    end
    function v = get.Drifts(W)
        v = {W.Drift1, W.Drift2};
    end    
    function set_Drifts(W, v)
        assert(iscell(v) && numel(v) == 2);
        W.set_Drift1(v{1});
        W.set_Drift2(v{2});
    end
    function set_Drift1(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Drift1 = W.enforce_class('Fit.D2.RT.BoundedCondEn.DriftEn', obj_or_name, {
            'dim_rel_W', 1
            });
        W.set_sub_from_props({'Drift1'});
    end
    function set_Drift2(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Drift2 = W.enforce_class('Fit.D2.RT.BoundedCondEn.DriftEn', obj_or_name, {
            'dim_rel_W', 2
            });
        W.set_sub_from_props({'Drift2'});
    end
end
methods (Static)
    function Drift = demo
        %%
        Drift = Fit.D2.RT.BoundedCondEn.Drift;
        disp(Drift);

        Drift.Data.set_path;
        tic;
        Drift.Data.load_data;
        toc;
        
        %%
        tic;
        drift_cond_t = Drift.get_drift_cond_t;
        toc;
        siz = size(drift_cond_t);
        disp(siz);
        assert(isequal(siz, [5001, 101, 2]));
        
        %%
        for i_dim = 1:2
            subplotRC(1,2,1,i_dim);
            imagesc(drift_cond_t(:,:,i_dim));
        end
    end
end
end