classdef PredEnSerMultSwitch < PredEnSerial
properties
    dur_per_dim_sec = [0.12, 0.12];
end
methods
    function W = PredEnSerMultSwitch(varargin)
        W.policy = 'SerMultSwitch';
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function [ch, rt, res, ens] = pred(W, en, bound)
        %%
        % prepare gain
        n_tr = size(en, 1);
        n_fr = size(en, 2);
        n_dim = size(en, 3);
        
        dim_1st = (rand(n_tr, 1) < W.p_dim1_1st) + 1;
        dur_per_dim_fr = round(W.dur_per_dim_sec * W.refresh_rate);
        gain_by_dim_1st = [ ...
            ones(1, dur_per_dim_fr(1)), zeros(1, dur_per_dim_fr(2))
            ones(1, dur_per_dim_fr(2)), zeros(1, dur_per_dim_fr(1))];
        gain_by_dim_1st = rep2fit(gain_by_dim_1st, ...
            [2, n_fr]);
        
        gain = zeros(size(en));
        for dim = 1:2
            incl = dim_1st==dim;
            gain(incl,:,dim) = repmat(gain_by_dim_1st(dim, :), ...
                [nnz(incl), 1]);
        end        
        [~, ~, res] = W.pred_ch_rt(en, gain, bound);
        for tr = 1:n_tr
            if res.td(tr,1) < res.td(tr,2) % dim 1 finished first
                gain(tr, (res.td(tr,1) + 1):end, 1) = 0;
                gain(tr, (res.td(tr,1) + 1):end, 2) = 1;
            else % dim 2 finished first
                gain(tr, (res.td(tr,2) + 1):end, 1) = 1;
                gain(tr, (res.td(tr,2) + 1):end, 2) = 0;
            end            
        end
        [ch, rt, res] = W.pred_ch_rt(en, gain, bound, 'noise', res.noise);

        if nargout >= 4
            ens = W.get_ens(rt, en);
        end
    end
end
end