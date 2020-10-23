classdef SigmaSq <  Fit.D1.Bounded.SigmaSqConst ...
                 & Fit.D2.Common.CommonWorkspace
    % Fit.D2.Common.SigmaSq
    %
    % Different from Fit.D2.Common.SigmaSq in that this contains
    % only one SigmaSq.
    % That's because SigmaSq depends on the number of conditions.
    % (e.g., in SigmaSqIndiv)
    %
    % 2015 YK wrote the initial version.
methods
    function W = SigmaSq(varargin)
        % SigmaSq('dim_rel', 1 or 2, ...)

        W.set_Data;
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
        obj_or_name = W.enforce_class('Fit.D2.Common.DataChRtPdf', ...
            obj_or_name);
        W.set_Data@Fit.Common.CommonWorkspace(obj_or_name);
    end    
end
end