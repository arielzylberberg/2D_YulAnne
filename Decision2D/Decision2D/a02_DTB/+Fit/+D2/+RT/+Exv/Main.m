classdef Main < Fit.D2.Common.Main
    % Fit.D2.RT.Exv.Main
    
    % 2016 YK wrote the initial version.
    
%% Properties - Settings
properties (Dependent)
%     % Defined in Common.Main
%     drift_kind
%     bound_kind
%     sigmaSq_kind
%     tnd_distrib
%     n_tnd
end
%% Properties - Intermediate variables
properties
    Dtb1
    Dtb2
    Td
    Tnd
    Miss
end
properties (Dependent)
    Dtbs
    Drifts
    Bounds
    SigmaSqs
end
%% Main
methods
    function W = Main(varargin)
        W.set_Dtbs;
        W.set_Td;
        W.set_Tnd;
        W.set_Miss;
        
        W.n_tnd = 4;
        W.tnd_kind = 'gamma';
        
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        W.init@Fit.D2.Common.Main(varargin{:});

        for dim = 1:2
            W.Drifts{dim}.customize_th_for_Data(dim);
        end
    end
    function pred(W)
        %% Calculate tds and unabs
        n_dim = 2;
        for dim = n_dim:-1:1
            tds{dim} = W.Dtbs{dim}.get_Td_pdf;
        end
        
        %% Calculate Td
        % TdExv takes maximum Td and boosts accuracy.
        [p_td_max, p_last] = W.Td.get_Td_pdf(tds);
        
        %% Adjust accuracy according to exhaustive accumulation
        for dim2nd = n_dim:-1:1
            dim1st = n_dim + 1 - dim2nd;
            
            nt = W.nt;
            n_cond = W.Data.nConds(dim1st);
            n_ch = 2;
            
            drift = rep2fit(W.Drifts{dim1st}.get_drift_cond_t, ...
                [n_cond, nt]);
            sigmaSq = rep2fit(W.SigmaSqs{dim1st}.get_sigmaSq_cond_t, ...
                [n_cond, nt]);
            bound = rep2fit(W.Bounds{dim1st}.get_bound_t_ch, ...
                [nt, n_ch]);
            dt = W.dt;
            
%             td_exv{dim} = W.Td.accumulate_unbounded_until_td_max( ...
%                 p_td_max, p_last, tds{dim_o}, dim, drift, bound, sigmaSq, dt);
            td_exv{dim2nd} = W.Td.accumulate_unbounded_until_td_max_vectorized( ...
                p_td_max, p_last, tds{dim1st}, dim2nd, ...
                drift, bound, sigmaSq, dt);
        end
        td = td_exv{1} + td_exv{2};
        W.Data.set_Td_pred_pdf(td);

        % DEBUG: Each condition should sum to 1.
        try
            assert_isequal_within(sums(td, [1, 4, 5]), 1, 1e-3, ...
                'relative_tol', false);
        catch err
            warning(err_msg(err));
        end
        
%         % Sanity check. 
%         % RT should look the same (if accuOnly = false) and 
%         % accuracy should be better.
%         W.Data.set_Td_pred_pdf(p_td_max); 
        
        %% Tnd and Miss
        W.Tnd.pred;
        W.Miss.pred;
    end
end
%% Object property kinds
methods
    function set_drift_kind(W, v)
        W.set_Drifts(v);
    end
    function v = get_drift_kind(W)
        f = @(v) strrep(bml.pkg.pkg2class(class(v)), 'Drift', '');
        v = {f(W.Drifts{1}), f(W.Drifts{2})};
        if isequal(v{1}, v{2})
            v = v{1};
        end
    end
    
    function set_bound_kind(W, v)
        W.set_Bounds(v);
    end
    function v = get_bound_kind(W)
        f = @(v) strrep(bml.pkg.pkg2class(class(v)), 'Bound', '');
        v = {f(W.Bounds{1}), f(W.Bounds{2})};
        if isequal(v{1}, v{2})
            v = v{1};
        end
    end 
    
    function set_sigmaSq_kind(W, v)
        W.set_SigmaSqs(v);
    end
    function v = get_sigmaSq_kind(W)
        f = @(v) strrep(bml.pkg.pkg2class(class(v)), 'SigmaSq', '');
        v = {f(W.SigmaSqs{1}), f(W.SigmaSqs{2})};
        if isequal(v{1}, v{2})
            v = v{1};
        end
    end
    
    function set_tnd_distrib(W, v)
        n_tnd = W.n_tnd;
        
        W.set_Tnd;
        W.Tnd.distrib = v;
        W.Tnd.n_Tnd = n_tnd;
        W.Tnd.init_params0;
    end
    function v = get_tnd_distrib(W)
        v = W.Tnd.distrib;
    end
    
    function set_n_tnd(W, v)
        distrib = W.tnd_kind;
        
        W.set_Tnd;
        W.Tnd.distrib = distrib;
        W.Tnd.n_Tnd = v;
        W.Tnd.init_params0;
    end
    function v = get_n_tnd(W)
        v = W.Tnd.n_Tnd;
    end
end
%% Drift
methods
    function Dtbs = get.Dtbs(W)
        Dtbs = {W.get_Dtb1, W.get_Dtb2};
    end
    function set.Dtbs(W, v)
        W.set_Dtbs(v);
    end
    function set_Dtbs(W, v)
        if exist('v', 'var')
            assert(iscell(v) && numel(v) == 2);
            W.set_Dtb1(v{1});
            W.set_Dtb2(v{2});
        else
            W.set_Dtb1;
            W.set_Dtb2;
        end
    end
    function set_Dtb1(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Dtb1 = W.enforce_class('Fit.D2.Bounded.Dtb1D', obj_or_name, ...
            {'dim_rel_W', 1});
        W.set_sub_from_props({'Dtb1'});
    end
    function set_Dtb2(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Dtb2 = W.enforce_class('Fit.D2.Bounded.Dtb1D', obj_or_name, ...
            {'dim_rel_W', 2});
        W.set_sub_from_props({'Dtb2'});
    end
    function set_Td(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Exv'; end
        W.Td = W.enforce_class('Fit.D2.RT.Exv.Td', obj_or_name);
        W.set_sub_from_props({'Td'});
    end
    function v = get_Dtb1(W)
        v = W.Dtb1;
    end
    function v = get_Dtb2(W)
        v = W.Dtb2;
    end
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
        v = {W.Dtb1.Drift, W.Dtb2.Drift};
    end
    function set_Drifts(W, v)
        if exist('v', 'var')
            if ~iscell(v), v = {v}; end
            if numel(v) == 1
                v = {v{1}, bml.oop.deep_copy_safe(v{1})};
            else
                assert(numel(v) == 2);
            end

            W.set_Drift1(v{1});
            W.set_Drift2(v{2});
        else
            W.set_Drift1;
            W.set_Drift2;
        end
    end
    
    function set_Drift1(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.Dtb1.set_Drift(obj_or_name);
    end
    function set_Drift2(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.Dtb2.set_Drift(obj_or_name);
    end
    
    function v = get_cond_bias(W)
        v = [W.Dtb1.Drift.th.bias, W.Dtb2.Drift.th.bias];
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
        v = {W.Dtb1.Bound, W.Dtb2.Bound};
    end    
    function set_Bounds(W, v)
        if exist('v', 'var')
            if ~iscell(v), v = {v}; end
            if numel(v) == 1
                v = {v{1}, bml.oop.deep_copy_safe(v{1})};
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
        if nargin < 2, obj_or_name = 'Const'; end
        W.Dtb1.set_Bound(obj_or_name);
    end
    function set_Bound2(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.Dtb2.set_Bound(obj_or_name);
    end        
end
%% SigmaSq
methods
    function v = get_SigmaSq(W, dim)
        v = W.SigmaSqs{dim};
    end
    function set.SigmaSqs(W, v)
        W.set_SigmaSqs(v);
    end
    function v = get.SigmaSqs(W)
        v = {W.Dtb1.SigmaSq, W.Dtb2.SigmaSq};
    end    
    function set_SigmaSqs(W, v)
        if exist('v', 'var')
            if ~iscell(v), v = {v}; end
            if numel(v) == 1
                v = {v{1}, bml.oop.deep_copy_safe(v{1})};
            else
                assert(numel(v) == 2);
            end

            W.set_SigmaSq1(v{1});
            W.set_SigmaSq2(v{2});
        else
            W.set_SigmaSq1;
            W.set_SigmaSq2;
        end
    end
    function set_SigmaSq1(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.Dtb1.set_SigmaSq(obj_or_name);
    end
    function set_SigmaSq2(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.Dtb2.set_SigmaSq(obj_or_name);
    end        
end
%% Tnd, Miss
methods
    function set_Tnd(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Tnd = W.enforce_class('Fit.D2.Common.Tnd', obj_or_name);
        W.set_sub_from_props({'Tnd'});
    end
    function set_Miss(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Miss = W.enforce_class('Fit.D2.Common.Miss', obj_or_name);
        W.set_sub_from_props({'Miss'});
    end
end
end