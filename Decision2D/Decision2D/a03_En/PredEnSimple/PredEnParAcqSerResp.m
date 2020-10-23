classdef PredEnParAcqSerResp < PredEnSerial
methods
    function W = PredEnParAcqSerResp(varargin)
        W.policy = 'ParAcqSerResp';
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function [ch, rt, res, ens] = pred(W, en, bound)
        %%
        [ch, ~, res] = W.pred@PredEnParallel(en, bound);
        n_fr = size(en, 2);
        rt = min(sum(res.td, 2), n_fr);
        if nargout >= 4
            ens = W.get_ens(rt, en);
        end
    end
end
end