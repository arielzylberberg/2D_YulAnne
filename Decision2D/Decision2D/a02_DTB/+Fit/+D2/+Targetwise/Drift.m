classdef Drift < Fit.D2.Common.Drift
    % Fit.D2.Targetwise.Drift
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
    function drift_cond_t = get_drift_cond_t(W, ch1, ch2)
        % drift_cond_t = get_drift_cond_t(W, ch1=2, ch2=2)
        %
        % drift_cond_t : nCond1 x nCond2 x nt
        % Different from Fit.D1.Bounded.Drift
        
        if ~exist('ch1', 'var')
            ch1 = 2; 
        else
            assert(isscalar(ch1) && any(ch1 == [1 2]));
        end
        if ~exist('ch2', 'var')
            ch2 = 2; 
        else
            assert(isscalar(ch2) && any(ch2 == [1 2]));
        end
        
        drift_cond_t1 = W.Drift1.get_drift_cond_t; % nCond1 x nt
        drift_cond_t2 = W.Drift2.get_drift_cond_t; % nCond2 x nt
        
        if ch1 == 1
            drift_cond_t1 = -drift_cond_t1;
        end
        if ch2 == 1
            drift_cond_t2 = -drift_cond_t2;
        end
        
        drift_cond_t1 = permute(drift_cond_t1, [1 3 2]); % nCond1 x 1 x nt
        drift_cond_t2 = permute(drift_cond_t2, [3 1 2]); % 1 x nCond2 x nt
        
        drift_cond_t = bsxfun(@plus, drift_cond_t1, drift_cond_t2);
    end
    %% Get/Set objects
    function set.Drifts(W, v)
        assert(iscell(v) && numel(v) == 2);
        W.set_Drift1(v{1});
        W.set_Drift2(v{2});
    end
    function v = get.Drifts(W)
        v = {W.Drift1, W.Drift2};
    end    
    function set_Drift1(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.Drift1 = W.enforce_class('Fit.D2.Common.Drift', obj_or_name, {
            'dim_rel_W', 1
            });
        W.set_sub_from_props({'Drift1'});
    end
    function set_Drift2(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.Drift2 = W.enforce_class('Fit.D2.Common.Drift', obj_or_name, {
            'dim_rel_W', 2
            });
        W.set_sub_from_props({'Drift2'});
    end
end
end