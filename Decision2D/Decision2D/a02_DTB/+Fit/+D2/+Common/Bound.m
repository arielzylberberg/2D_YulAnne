classdef Bound < Fit.D2.Common.CommonWorkspace
    % Fit.D2.Common.Bound
    %
    % Different from Fit.D2.Common.Drift in that this contains
    % two pairs of Bounds.
    % That's because Bound doesn't depend on the number of conditions.
    %
    % 2015 YK wrote the initial version.
properties (SetAccess = protected)
    Bound1
    Bound2
end
properties (Dependent)
    Bounds
    dim_bound_t_ch
end
methods
    function W = Bound
        subs = {'Bound1', 'Bound2'};        
        W.add_deep_copy(subs);
        
        W.set_Data;
        W.set_Bound1;
        W.set_Bound2;
    end
    %% Predictions
    function bound_t_ch = get_bound_t_ch(W, varargin)
        % bound_t_ch : nt x ch x dim
        % Different from Fit.D1.Bounded.Bound or Exhaustive.Bound:
        % ch x dim rather than ch x ch.
        
        bound1 = W.Bound1.get_bound_t_ch(varargin{:}); % nt x 2
        bound2 = W.Bound2.get_bound_t_ch(varargin{:}); % nt x 2
        
        bound_t_ch = cat(3, bound1, bound2);
    end
    function d = get.dim_bound_t_ch(W)
        d = varargin2S({
            't', 1
            'ch', 2
            'dim', 3
            });
    end
end
methods
    %% Get/Set objects
    function set.Bounds(W, v)
        W.set_Bounds(v);
    end
    function set_Bounds(W, v)
        if ~iscell(v), v = {v}; end
        if numel(v) == 1
            v = {v{1}, bml.oop.deep_copy_safe(v{1})};
        else
            assert(numel(v) == 2);
        end
        
        W.set_Bound1(v{1});
        W.set_Bound2(v{2});
    end
    function v = get.Bounds(W)
        v = W.get_Bounds;
    end
    function v = get_Bounds(W)
        v = {W.Bound1, W.Bound2};
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
%% Plot
methods 
    function plot(W)
        t = W.get_t;
        bound_t_ch = W.get_bound_t_ch;
        n_dim = size(bound_t_ch, W.dim_bound_t_ch.dim);
        
        for dim = 1:n_dim
            subplotRC(n_dim, 1, dim, 1);
            plot(t, bound_t_ch(:,:,dim));
            title(sprintf('Bound - Dim%d', dim));
            grid on;
        end
        xlabel('t (s)');
        ylabel('Ev (logit)');
    end
end
end