classdef DtbCalc < Fit.D2.Common.CommonWorkspace
    % Fit.D2.RT.BoundedCondEn.DtbCalc
    %
    % 2015 YK wrote the initial version.
properties (SetAccess = private)
    % Required, not expanded
    n_tr = [];
    
    % Required, expanded
    drift
    bound
    tnd_st
    
    % Not required, expanded
    sigmaSq_fac_bef_start = [0 0];
    sigmaSq_fac_together  = [1 1];
    drift_fac_together  = [1 0];
    
    % Not expanded
    n_dim = 2;
    n_ch = 2;
    use_gpu = false;
end
methods
    function W = DtbCalc(varargin)
        % W = DtbCalcSim(n_tr, drift, bound, tnd_st, ...)
        %
        % n_tr: scalar
        % drift: tr x t x dim
        % bound: t x ch x dim => expanded to tr x t x ch x dim
        % tnd_st: t x dim => expanded to tr x dim
        % 
        % Optional arguments:
        % sigma_alone = [1 1]; % No reason to change
        % sigmaSq_fac_bef_start = [0 0];
        % sigmaSq_fac_together  = [1 1];
        % drift_fac_together  = [1 0];
        % n_dim = 2;
        % n_ch = 2;
        % use_gpu = false;
        
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        required_props = {
            'n_tr', 'drift', 'bound', 'tnd_st'};
        S = varargin2S(varargin);
        assert(all(isfield(S, required_props)));
        
        % Copy fields in an appropriate order.
        % To change individual properties, call set_* methods directly.
        copyprops(W, S, 'props', required_props);
        S = rmfield(S, required_props);
        
        % Optional
        S = varargin2S(S, {
            % Not required, expanded
            'sigmaSq_fac_bef_start', [0 0];
            'sigmaSq_fac_together', [1 1];
            'drift_fac_together', [1 0];
            % Not expanded
            'n_dim', 2;
            'n_ch', 2;
            'use_gpu', false;
            });
        copyprops(W, S);
    end
end
%% Facade
methods
    function [td, ch, traj] = calc_dtb(W, varargin)
        W.init(varargin{:});
        [td, ch, traj] = W.get_pred_td_tr_t_ch;
    end
end
methods (Static)
    function [td, ch, traj, W] = calc_dtb_new(varargin)
        W = feval(my_class, varargin{:});
        [td, ch, traj] = W.get_pred_td_tr_t_ch;
    end
end
%% Calculation
methods
    function [td, ch, traj] = get_pred_td_tr_t_ch(W)
        % td: tr x 1
        % ch: tr x dim
        % traj: tr x dim x t
        
        error('Implement in subclasses!');
    end
end
%% Get/Set
methods
    %% Get/Set
    function set_n_tr(W, n_tr)
        assert(isscalar(n_tr) && isnumeric(n_tr) && n_tr >= 0);
        W.n_tr = n_tr;
    end
    function n_tr = get_n_tr(W)
        n_tr = W.n_tr;
    end
    
    function set_drift(W, v)
        % drift: tr x t x dim
        W.drift = v;
    end
    function v = get_drift(W)
        v = W.drift;

        n_tr = W.get_n_tr;
        nt = W.get_nt;
        n_dim = W.get_n_dim;     
        
        assert(isnumeric(v));
        v = rep2fit_strict(v, [n_tr, nt, n_dim]);
    end
    
    function set_bound(W, v)        
        % input: t x ch x dim
        W.bound = v;
    end
    function v = get_bound(W)
        v = W.bound;
        
        nt = W.get_nt;
        n_ch = W.get_n_ch;
        n_dim = W.get_n_dim;
        assert(isequal(size(v), [nt, n_ch, n_dim]));
        
        n_tr = W.get_n_tr;
        v = repmat(permute(v, [4, 1, 2, 3]), [n_tr, 1]);
    end
    
    function set_tnd_st(W, v)
        % input: nt x dim
        W.tnd_st = v;
    end
    function v = get_tnd_st(W)
        v0 = W.tnd_st;
        
        assert(isnumeric(v0));
        
        nt = W.get_nt;
        n_dim = W.get_n_dim;
        n_tr = W.get_n_tr;
        
        assert(isequal(size(v0), [nt, n_dim]));
        
        v = zeros(n_tr, n_dim);
        for i_dim = 1:n_dim
            v(:,i_dim) = randsample(nt, n_tr, true, v0(:, i_dim));
        end        
    end

    function set_sigmaSq_fac_bef_start(W, v)
        assert(isequal(size(v), [1, W.get_n_dim]) ...
            || isequal(size(v), [W.get_n_tr, W.get_n_dim]));
        W.sigmaSq_fac_bef_start = v;
    end
    function v = get_sigmaSq_fac_bef_start(W)
        v = W.sigmaSq_fac_bef_start;
        
        n_tr = W.get_n_tr;
        n_dim = W.get_n_dim;
        v = rep2fit(v, [n_tr, n_dim]);
    end

    function set_sigmaSq_fac_together(W, v)
        assert(isequal(size(v), [1, W.get_n_dim]) ...
            || isequal(size(v), [W.get_n_tr, W.get_n_dim]));
        W.sigmaSq_fac_together = v;
    end
    function v = get_sigmaSq_fac_together(W)
        v = W.sigmaSq_fac_together;
        
        n_tr = W.get_n_tr;
        n_dim = W.get_n_dim;
        v = rep2fit(v, [n_tr, n_dim]);
    end

    function set_drift_fac_together(W, v)
        assert(isequal(size(v), [1, W.get_n_dim]) ...
            || isequal(size(v), [W.get_n_tr, W.get_n_dim]));
        W.drift_fac_together = v;
    end
    function v = get_drift_fac_together(W)
        v = W.drift_fac_together;
        
        n_tr = W.get_n_tr;
        n_dim = W.get_n_dim;
        v = rep2fit(v, [n_tr, n_dim]);
    end

    % Not expanded
    function set_n_dim(W, v)
        W.n_dim = v;
    end
    function v = get_n_dim(W)
        v = W.n_dim;
    end

    function set_n_ch(W, v)
        W.n_ch = v;
    end
    function v = get_n_ch(W)
        v = W.n_ch;
    end

    function set_use_gpu(W, v)
        W.use_gpu = v;
    end
    function v = get_use_gpu(W)
        v = W.use_gpu;
    end
end
end