classdef Dtb < Fit.D2.Bounded.Dtb
    % Fit.D2.Inh.Dtb
    %
    % 2015 YK wrote the initial version.
properties (Dependent)
    sigmaSq_fac_bef_start % dim x 1
    sigmaSq_fac_together  % dim x first_dim
    drift_fac_together  % dim x first_dim
end
properties (SetAccess = protected)
    Drift1
    Drift2
    Bound1
    Bound2    
end
properties
    KBRatio1 = [];
    KBRatio2 = [];
end
properties (Dependent)
    Drifts
    Bounds
end
methods
    function W = Dtb
        W.set_Data;
        W.add_params0;
    end
    function add_params0(W)
        W.add_params({
            {'p_dim1_1st', 0.5, 0, 1}
            ...
            {'sigmaSq_fac_bef_start_dim1', 0, 0, 0}
            {'sigmaSq_fac_bef_start_dim2', 0, 0, 0}
            ...
            % dimA_B : sigma of dimA when dimB is prioritized.
            % Without loss of generality, when prioritized, it is fixed to 1.
            % Since noise level is relative, sigmaSq can be > 1.
            {'sigmaSq_fac_together_dim1_1', 1, 1, 1}
            {'sigmaSq_fac_together_dim1_2', 1, 0.16, 2}
            {'sigmaSq_fac_together_dim2_1', 1, 0.16, 2}
            {'sigmaSq_fac_together_dim2_2', 1, 1, 1}
            ...
            % dimA_B : sigma of dimA when dimB is prioritized.
            % Without loss of generality, when prioritized, it is fixed to 1.
            {'drift_fac_together_dim1_1', 1, 1, 1}
            {'drift_fac_together_dim1_2', 1, 0, 1}
            {'drift_fac_together_dim2_1', 1, 0, 1}
            {'drift_fac_together_dim2_2', 1, 1, 1}
%             {'sigmaSq_fac_bef_start_dim1', 0, 0, 0}
%             {'sigmaSq_fac_bef_start_dim2', 0, 0, 0}
%             ...
%             {'sigmaSq_fac_together_dim1_1', 1, 1, 1}
%             {'sigmaSq_fac_together_dim1_2', 1, 1, 1}
%             {'sigmaSq_fac_together_dim2_1', 1, 1, 1}
%             {'sigmaSq_fac_together_dim2_2', 1, 1, 1}
%             ...
%             {'drift_fac_together_dim1_1', 1, 1, 1}
%             {'drift_fac_together_dim1_2', 1, 1, 1}
%             {'drift_fac_together_dim2_1', 1, 1, 1}
%             {'drift_fac_together_dim2_2', 1, 1, 1}
            });
    end
    function constrain_fano_unit(W, dim, dim_1st, fano)
        if fano < inf
            ssq = sprintf('sigmaSq_fac_together_dim%d_%d', dim, dim_1st);
            drift = sprintf('drift_fac_together_dim%d_%d', dim, dim_1st);
            
            W.add_constraints({
                {'A', {drift, ssq}, {[1, -fano], 0}}
                });
        end
    end
end
%% Main calculation
methods
    function pred(W)
        warning('Modify in subclasses!');
    end
end
%% Dtb parameters
methods
    function drift_cond_t = get_drift_cond_t(W)
        % drift_cond_t: n_tr x nt x n_dim
        drift_cond_t = W.Drift.get_drift_cond_t;
        assert(isequal(size(drift_cond_t), W.get_drift_cond_t_size));
    end
    function size_ = get_drift_cond_t_size(W)
        size_ = [W.Data.get_n_tr, W.Time.get_nt, W.Data.get_n_dim];
    end
    function bound_t_ch = get_bound_t_ch(W)
        % bound_t_ch: nt x ch x dim
        bound_t_ch = W.Bound.get_bound_t_ch;
        assert(isequal(size(bound_t_ch), W.get_bound_t_ch_size));
    end
    function size_ = get_bound_t_ch_size(W)
        size_ = [W.Time.get_nt, 2, W.Data.get_n_dim];
    end
    function set_sigma(W, sigma)
        assert(isscalar(sigma) && isnumeric(sigma) && sigma > 0);
        W.sigma = sigma;
    end
    function sigma = get_sigma(W)
        sigma = W.sigma;
    end
end
%% Get/Set - fit parameters
methods
    function v = get_dim_1st_incl(W)
        p_dim1_1st = W.get_th_('p_dim1_1st');
        if p_dim1_1st == 1
            v = 1;
        elseif p_dim1_1st == 0
            v = 2;
        else
            v = [2 1];
        end
    end
    function v = get_sigmaSq_fac_bef_start(W)
        th = W.th;
        v = [
            th.sigmaSq_fac_bef_start_dim1
            th.sigmaSq_fac_bef_start_dim2
            ];
    end
    function v = get_sigmaSq_fac_together(W, dim, dim_1st)
        % dimK_M : factor of K-th dim when M-th dim is prioritized first.
        th = W.th;
        v = [
            th.sigmaSq_fac_together_dim1_1, th.sigmaSq_fac_together_dim1_2
            th.sigmaSq_fac_together_dim2_1, th.sigmaSq_fac_together_dim2_2
            ];
        
        if nargin >= 2
            v = v(dim, dim_1st);
        end
    end
    function v = get_drift_fac_together(W, dim, dim_1st)
        % dimK_M : factor of K-th dim when M-th dim is prioritized first.
        th = W.th;
        v = [
            th.drift_fac_together_dim1_1, th.drift_fac_together_dim1_2
            th.drift_fac_together_dim2_1, th.drift_fac_together_dim2_2
            ];
        
        if nargin >= 2
            v = v(dim, dim_1st);
        end
    end
    function v = get.sigmaSq_fac_bef_start(W)
        v = W.get_sigmaSq_fac_bef_start;
    end
    function v = get.sigmaSq_fac_together(W)
        v = W.get_sigmaSq_fac_together;
    end
    function v = get.drift_fac_together(W)
        v = W.get_drift_fac_together;
    end
end
%% Drift
methods
    function v = get_Drift(W, dim)
        v = W.Drifts{dim};
    end
    
    function set.Drifts(W, v)
        W.set_Drifts(v);
    end
    function v = get.Drifts(W)
        v = W.get_Drifts;
    end
    
    function v = get_Drifts(W)
        v = {W.Drift1, W.Drift2};
    end
    function set_Drifts(W, v)
        if exist('v', 'var')
            if iscell(v)
                if isscalar(v)
                    v = rep_deep_copy(v, [1, 2]);
                end
                assert(numel(v) == 2);
                W.set_Drift1(v{1});
                W.set_Drift2(v{2});
            elseif ischar(v)
                W.set_Drift1(v);
                W.set_Drift2(v);
            end
        else
            W.set_Drift1;
            W.set_Drift2;
        end
    end
    
    function set_Drift1(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Drift1 = W.enforce_class('Fit.D2.Common.Drift', obj_or_name, {
            'dim_rel_W', 1
            });
        W.set_sub_from_props({'Drift1'});
        W.Drift1.customize_th_for_Data(1);
    end
    function set_Drift2(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Drift2 = W.enforce_class('Fit.D2.Common.Drift', obj_or_name, {
            'dim_rel_W', 2
            });
        W.set_sub_from_props({'Drift2'});
        W.Drift2.customize_th_for_Data(2);
    end
end
%% Bound
methods
    function v = get_Bound(W, dim)
        v = W.Bounds{dim};
    end
    function set.Bounds(W, v)
        W.set_Bounds(v);
    end
    function v = get.Bounds(W)
        v = {W.Bound1, W.Bound2};
    end    
    function set_Bounds(W, v)
        if exist('v', 'var')
            if ~iscell(v), v = {v}; end
            if numel(v) == 1
                v = {v{1}, deep_copy(v{1})};
            else
                assert(numel(v) == 2);
            end

            W.set_Bound1(v{1});
            W.set_Bound2(v{2});
        else
            W.set_Bound1;
            W.set_Bound2;
        end
    end
    function set_Bound1(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Bound1 = W.enforce_class('Fit.D1.Bounded.Bound', obj_or_name);
        W.set_sub_from_props({'Bound1'});
    end
    function set_Bound2(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Bound2 = W.enforce_class('Fit.D1.Bounded.Bound', obj_or_name);
        W.set_sub_from_props({'Bound2'});
    end        
end
%% KBRatio
methods
    function set_KBRatios(W, KB)
        if nargin < 2
            KB = {Fit.D2.Common.KBRatio, Fit.D2.Common.KBRatio};
        end
        for ii = 1:2
            W.set_KBRatio(KB{ii}, ii);
        end
    end
    function set_KBRatio(W, KB, ix)
        % set_KBRatio(W, KB, ix)
        assert(isa(KB, 'Fit.D2.Common.KBRatio'));
        assert(isnumeric(ix));
        assert(isscalar(ix));
        
        kb_name = sprintf('KBRatio%d', ix);
        drift_name = sprintf('Drift%d', ix);
        bound_name = sprintf('Bound%d', ix);
        
        W.(kb_name) = KB;
        KB.set_Drift(W.(drift_name));
        KB.set_Bound(W.(bound_name));
        
        W.set_sub_from_props({kb_name});
        W.(kb_name).customize_th_for_Data(ix);
    end
    function calc_KBRatio(W)
        if ~isempty(W.KBRatio1), W.KBRatio1.pred; end
        if ~isempty(W.KBRatio2), W.KBRatio2.pred; end
    end
end
%% KBRatio - nonlcon
% Use nonlcon rather than KBRatio objects
properties
    k_b_prod_ub = [100, 40];
    k_b_prod_lb = [2, 2];
    k_b_ratio_ub = [120, 1];
    k_b_ratio_lb = [120, 1];
end
methods
    function set_KBRatio_nonlcon(W)
%         for ii = 1:2
%             W.set_KBRatio_nonlcon_unit(ii);
%         end
%     end
%     function set_KBRatio_nonlcon_unit(W, ix)
        W.add_constraints({
            {'c', {'Drift1__k', 'Bound1__b'}, ...
                {@(v) v(1) * v(2) - W.k_b_prod_ub(1) .^ 2}}
            {'c', {'Drift1__k', 'Bound1__b'}, ...
                {@(v) W.k_b_prod_lb(1) .^ 2 - v(1) * v(2)}}
            {'c', {'Drift2__k', 'Bound2__b'}, ...
                {@(v) v(1) * v(2) - W.k_b_prod_ub(2) .^ 2}}
            {'c', {'Drift2__k', 'Bound2__b'}, ...
                {@(v) W.k_b_prod_lb(2) .^ 2 - v(1) * v(2)}}
            });
    end
end
end