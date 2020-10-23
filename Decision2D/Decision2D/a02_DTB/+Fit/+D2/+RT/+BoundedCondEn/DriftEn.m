classdef DriftEn < Fit.D1.Bounded.DriftConst & Fit.D2.Common.CommonWorkspace
    %
    % 2015 YK wrote the initial version.
methods
    function W = DriftEn(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
        
        W.kind = 'en';        
        W.init_W0;
    end
    function init(W, varargin)
        S = varargin2S(varargin, {
            'dim_rel_W', []
            });
        if ~isempty(S.dim_rel_W)
            W.set_dim_rel_W(S.dim_rel_W);
        end
    end
    function drift_vec = get_drift_vec(W)
        % drift_vec: nCond x 1 vector.
        % Used for plotting.
        drift_vec = nanmean(W.get_drift_cond_t, 2); % Doesn't make sense here.
        dCond = W.Data.get_dCond;
        drift_vec = accumarray(dCond, drift_vec);
    end
    function drift_cond_t = get_drift_cond_t(W)
        % Enforce nCond x nt form
        v = W.get_cond_t;
        drift_cond_t = W.cond2drift(v);
    end
    function v = get_cond_t(W)
        % Enforce nCond x nt form
        En = W.Data.get_En(W.get_dim_rel_W);
        v = En.get_ts_mat;
    end
    function guess_from_glmfit(W)
        error('Not implemented yet!');
    end
    function varargout = plot(W, varargin)
        if isempty(varargin), varargin = {'o-'}; end
        [varargout{1:nargout}] = plot( ...
            W.get_conds_rel, W.get_drift_vec, varargin{:});
    end
    %% Internal
    function set_conds(W, conds)
        error('Not defined for En!');
    end
    function conds = get_conds_rel(W)
        conds = W.Data.get_conds;
        conds = conds{W.get_dim_rel_W};
    end
    %% Get/Set
    function set_Data(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        obj_or_name = ...
            W.enforce_class('Fit.D2.Common.DataChRtPdfEn', ...
                obj_or_name);
        W.set_Data@Fit.Common.CommonWorkspace(obj_or_name);
    end
end
methods (Static)
    function W = demo
        W = Fit.D1.CondEn.Drift;
        W.set_Data(Fit.D1.Common.DataChRtPdfEn.demo);
        W.plot;
    end
end
end