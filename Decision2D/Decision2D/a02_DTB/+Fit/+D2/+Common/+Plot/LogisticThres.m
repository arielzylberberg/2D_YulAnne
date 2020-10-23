classdef LogisticThres < Fit.D2.Common.Plot.Logistic
methods
    function Plt = LogisticThres(varargin)
        if nargin > 0
            Plt.init(varargin{:});
            Plt.plot;
        end
    end
    function y = get_y(Plt)
        res = Plt.res;
        n_cond_irr = numel(res);
        
        y = zeros(n_cond_irr, 1);
        for ii = 1:n_cond_irr
            y(ii) = bml.stat.logit2thres(res{ii}.b);
        end        
    end
    function [lb, ub] = get_bnd(Plt)
        res = Plt.res;
        n_cond_irr = numel(res);
        
        lb = zeros(n_cond_irr, 1);
        ub = zeros(n_cond_irr, 1);
        
        for ii = 1:n_cond_irr
            bias = res{ii}.b(1);
        
            % When b is low, threshold is high, and vice versa.
            b_lb = res{ii}.b(2) - res{ii}.se(2);
            ub(ii) = bml.stat.logit2thres([bias, b_lb]);
            
            b_ub = res{ii}.b(2) + res{ii}.se(2);
            lb(ii) = bml.stat.logit2thres([bias, b_ub]);
        end
    end
    function ylabel(~)
        ylabel('Threshold');
    end
end
end