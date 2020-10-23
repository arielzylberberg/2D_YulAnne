classdef DtbKBRatioPairDrift < Fit.D2.Bounded.Dtb
    % Fit.D2.Bounded.DtbKBRatioPairDrift
    %
    % Decorates Dtb that has Drift1,2 and Bound1,2
    %
    % 2015 YK wrote the initial version.
properties (SetAccess = protected)
    Dtb
end
methods
    function W = DtbKBRatioPairDrift(varargin)
        W.add_deep_copy({'Dtb'});
        
        W.set_Data;
        W.set_Dtb;
        W.add_params0;
        
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function add_params0(W)
        W.Dtb.add_params0;
        W.replace_kb_w_ratio;
    end
    function pred(W)
        W.get_kb_from_ratio;
        W.Dtb.pred;
    end
    
    %%
    function set_Dtb(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Dtb = W.enforce_class('Fit.D2.Bounded.Dtb', obj_or_name);
%         W.Dtb = W.enforce_class('Fit.D2.Inh.DtbDensity', obj_or_name);
        W.set_sub_from_props({'Dtb'});
    end
    
    %%
    function replace_kb_w_ratio(W)
        Dtb = W.Dtb;
        
        n_dim = 2;
        for dim = 1:n_dim
            drift = sprintf('Drift%d', dim);
            bound = sprintf('Bound%d', dim);
            
            Dtb.(drift).remove_params({'k'});
            Dtb.(bound).remove_params({'b'});
            Dtb.add_params({
                {str_con('k_b_prod', dim), 6, 1.5, 90}
                {str_con('k_b_ratio', dim), 50, 2, 120}
                });
        end
    end
    function get_kb_from_ratio(W)
        Dtb = W.Dtb;
        
        n_dim = 2;
        for dim = 1:n_dim
            kbprod = str_con('k_b_prod', dim);
            kbratio = str_con('k_b_ratio', dim);
            drift = sprintf('Drift%d', dim);
            bound = sprintf('Bound%d', dim);
            
            k = sqrt(Dtb.th.(kbprod) .* Dtb.th.(kbratio));
            b = sqrt(Dtb.th.(kbprod) ./ Dtb.th.(kbratio));
            
            Dtb.(drift).add_th_forced('k', k);
            Dtb.(bound).add_th_forced('b', b);
        end
    end
end
end