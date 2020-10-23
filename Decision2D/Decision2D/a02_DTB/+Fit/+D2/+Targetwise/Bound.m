classdef Bound < Fit.D2.Common.Bound
    %
    % 2015 YK wrote the initial version.
methods
    %% Predictions
    function bound_t_ch = get_bound_t_ch(W)
        % bound_t_ch : nt x ch1 x ch2
        % Different from Fit.D2.Common.Bound:
        % (1) ch1 x ch2 rather than ch x dim.
        % (2) All values are positive.
        
        bound1 = abs(W.Bound1.get_bound_t_ch); % nt x 2
        bound2 = abs(W.Bound2.get_bound_t_ch); % nt x 2
        
        bound1 = permute(bound1, [1 2 3]); % nt x 2 x 1
        bound2 = permute(bound2, [1 3 2]); % nt x 1 x 2
        
        bound_t_ch = bsxfun(@plus, bound1, bound2);
    end
end
end