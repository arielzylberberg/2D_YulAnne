classdef PredEnSimple < FitWorkspace
properties
    policy
    refresh_rate = 75;
end
methods
    function W = PredEnSimple(varargin)
        % Default implementation is parallel, since it's the simplest.
        W.policy = 'parallel';
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        varargin2props(W, varargin);
    end
    function [ch, rt, res, ens] = pred(W, en, bound, varargin)
        gain = ones(size(en));
        [ch, rt, res, ens] = W.pred_ch_rt(en, gain, bound, varargin{:});
    end
    function [ch, rt, res, ens] = pred_ch_rt(...
            W, en, gain, bound, varargin)
        % en(tr, fr, dim)
        % use(tr, fr, dim)
        % bound(tr, fr, dim)
        %
        % ch(tr, dim)
        % rt(tr, 1)
        % gain(tr, fr, dim)
        % tnd(tr, 1)
        
        S = varargin2S(varargin, {
            'tnd_mean_sec', 0.3
            'tnd_std_sec', 0.1
            'diffusion_per_sec', 1
            'noise', []
            });
        
        n_dim = 2;
        n_tr = size(en, 1);
        n_fr = size(en, 2);        
        
        if isempty(S.noise)
            noise = normrnd(0, sqrt(S.diffusion_per_sec / W.refresh_rate), ...
                [n_tr, n_fr, n_dim]);
        else
            noise = S.noise;
        end
        en = en + noise;
        
        en_used = en .* gain;
        cum_en = cumsum(en_used, 2);
                
        crossed_bound = abs(cum_en) >= bound;
        td = permute(find_ndim(crossed_bound, 2), [1, 3, 2]);
        td(td == 0) = n_fr;        
        rt = max(td, [], 2);        
        
        %%
        ch = zeros(n_tr, n_dim);
        for dim = 1:n_dim
            ix = sub2ind([n_tr, n_fr, n_dim], ...
                1:n_tr, td(:, dim)', ...
                zeros(1, n_tr) + dim);
            ch(:, dim) = sign(cum_en(ix)) / 2 + 1.5;
        end
        rand_ch = (rand(size(ch)) > 0.5) + 1;
        ch(ch == 1.5) = rand_ch(ch == 1.5);
        
        %%
        for tr = 1:n_tr
            for dim = 1:n_dim
                gain(tr, (td(tr, dim) + 1):end, dim) = 0;
            end
        end
        
        %%
%         tnd_mean_fr = round(S.tnd_mean_sec * W.refresh_rate);
%         tnd_std_fr = round(S.tnd_std_sec * W.refresh_rate);

        tnd = W.get_tnd(S.tnd_mean_sec, S.tnd_std_sec, n_tr);
        rt = rt + tnd;        
        rt = max(min(rt, n_fr), 1);
        
        res = packStruct(td, gain, bound, tnd, noise);        
        if nargout >= 4
            ens = W.get_ens(rt, en);
        end
    end
    function tnd = get_tnd(W, mean_tnd, std_tnd, n_tr)
        tnd = round(gamrnd_ms(mean_tnd, std_tnd, [n_tr, 1]) ...
            * W.refresh_rate);
    end
    function ens = get_ens(W, rt, en)
        n_dim = size(en, 3);
        n_tr = size(rt, 1);
        for tr = 1:n_tr
            en(tr, (rt(tr) + 1):end, :) = nan;
        end

        ens = cell(1, n_dim);
        for dim = 1:n_dim
            ens{dim} = row2cell2(en(:,:,dim));
        end
    end
end
end