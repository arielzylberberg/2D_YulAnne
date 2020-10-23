classdef KBRatio < Fit.Common.KBRatio & Fit.D2.Common.CommonWorkspace
    % Fit.D2.Common.KBRatio
    %
    % Uses choices on dim_rel_W to set th0, lb, ub of k_b_prod.
    %
    % 2015 YK wrote the initial version.
properties (SetAccess = protected)
    k_b_prod_range = [1 / 1.2, 1.2];
end
methods
    function W = KBRatio(varargin)
        bml.oop.varargin2props(W, varargin, true);
        
        W.add_deep_copy({'Drift', 'Bound'});
        W.add_params0;
    end
    function customize_th_for_Data(W, dim_rel_W)
        % customize_th_for_Data(W, dim_rel_W)
        W.set_dim_rel_W(dim_rel_W);
        W.calc_k_b_prod;
    end
    function calc_k_b_prod(W)
        W.set_k_b_prod(W.get_k_b_prod);
    end
    function set_k_b_prod(W, k_b_prod)
        W.th.k_b_prod = k_b_prod;
        W.th0.k_b_prod = k_b_prod;
        W.lb.k_b_prod = k_b_prod * W.k_b_prod_range(1);
        W.ub.k_b_prod = k_b_prod * W.k_b_prod_range(2);
    end
    function [k_b_prod, se, res] = get_k_b_prod(W)
        assert(W.Data.loaded, ...
            'Call W.Data.load_data before set_KBRatios!');
        
        ch = W.get_ch_rel_W;
        cond = W.get_cond_rel_W;
        
        res = glmwrap(cond, ch == 2, 'binomial');
        k_b_prod = res.b(2) / 2;
        se = res.se(2);
    end
    function v = get_ch_rel_W(W)
        v = W.Data.get_ch;
        v = v(:, W.get_dim_rel_W);
    end
    function v = get_cond_rel_W(W)
        v = W.Data.get_cond;
        v = v(:, W.get_dim_rel_W);
    end
    function set_k_b_prod_range(W, v)
        % set_k_b_prod_range(W, v)
        %
        % v : scalar or 2-vector.
        assert(isnumeric(v));
        
        if isscalar(v)
            assert(v >= 1);
            W.k_b_prod_range = [1 / v, v];
        else
            assert(numel(v) == 2);
            assert(v(1) <= 1);
            assert(v(2) >= 1);
            W.k_b_prod_range = v;
        end
    end
    
%     function add_params0(W)
%         W.add_params({
%             {'k_b_prod', 20, 1.5, 90}
%             {'k_b_ratio', 20, 2, 120}
%             });
%     end
%     function pred(W)
%         assert(~isempty(W.Drift));
%         assert(~isempty(W.Bound));
%         
%         k_b_prod = W.th.k_b_prod;
%         k_b_ratio = W.th.k_b_ratio;
%         
%         k = sqrt(k_b_prod * k_b_ratio);
%         b = sqrt(k_b_prod / k_b_ratio);
%         
%         W.Drift.add_th_forced('k', k);
%         W.Bound.add_th_forced('b', b);
%     end
%     function set_Drift(W, Drift)
%         assert(isempty(Drift) || isa(Drift, 'FitParams'));
% %         assert(any(strcmp('k', Drift.get_names)));
%         assert(~isempty(strfind(class(Drift), 'Drift')));
%         W.Drift = Drift;
%         
%         try
%             Drift.add_th_forced('k', Drift.th0.k);
%         catch
%             Drift.add_th_forced('k', 20);
%         end
%         Drift.remove_params({'k'});
%     end
%     function set_Bound(W, Bound)
%         assert(isempty(Bound) || isa(Bound, 'FitParams'));
% %         assert(any(strcmp('b', Bound.get_names)));
%         assert(~isempty(strfind(class(Bound), 'Bound')));
%         W.Bound = Bound;
%         try
%             Bound.add_th_forced('b', Bound.th0.b);
%         catch
%             Bound.add_th_forced('b', 1);
%         end
%         Bound.remove_params({'b'});
%     end
end
end