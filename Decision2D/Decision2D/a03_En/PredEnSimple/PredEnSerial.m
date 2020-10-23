classdef PredEnSerial < PredEnParallel
properties
    p_dim1_1st = 0.5;
end
methods
    function W = PredEnSerial(varargin)
        W.policy = 'serial';
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function [ch, rt, res, ens] = pred(W, en, bound)
        %%
        [~, ~, res] = W.pred@PredEnParallel(en, bound);
        gain = res.gain;
        td = res.td;
        
        n_tr = size(en, 1);
        dim_1st = (rand(n_tr, 1) < W.p_dim1_1st) + 1;
        
        % Make dim2's gain zero until td1
        n_dim = size(en, 3);
        for tr = 1:n_tr
            dim1 = dim_1st(tr);
            dim2 = n_dim + 1 - dim1;
            gain(tr, 1:td(tr, dim1), dim2) = 0;
            gain(tr, (td(tr, dim1) + 1):end, dim2) = 1;
        end
        [ch, rt, res] ...
            = W.pred_ch_rt(en, gain, bound, 'noise', res.noise);

        if nargout >= 4
            ens = W.get_ens(rt, en);
        end
    end
end
end