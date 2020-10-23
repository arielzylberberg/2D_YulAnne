classdef Drift < Fit.D2.Common.CommonWorkspace
    % Fit.D2.Common.Drift
    %
    % Different from Fit.D2.Common.Drift in that this contains
    % only one Drift.
    % That's because Drift depends on the number of conditions.
    % (e.g., in DriftIndiv)
    %
    % 2015 YK wrote the initial version.
methods
    function W = Drift(varargin)
        % Drift('dim_rel', 1 or 2, ...)
        S = varargin2S(varargin, {
            'dim_rel_W', []
            });
        W.set_Data;
        if ~isempty(S.dim_rel_W)
            W.set_dim_rel_W(S.dim_rel_W);
        end
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function conds_rel = get_conds_rel(W)
        conds = W.Data.get_conds;
        dim_rel_W = W.get_dim_rel_W;
        if isempty(dim_rel_W)
            conds_rel = [];
        else
            conds_rel = conds{dim_rel_W};
        end
    end
    function set_Data(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        obj_or_name = W.enforce_class('Fit.D2.Common.DataChRtPdf', obj_or_name);
        W.set_Data@Fit.Common.CommonWorkspace(obj_or_name);
    end    
end
end