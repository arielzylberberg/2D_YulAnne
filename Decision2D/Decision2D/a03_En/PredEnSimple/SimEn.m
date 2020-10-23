classdef SimEn < FitWorkspace
properties
    refresh_rate = 75;
    t_max = 5;
    n_tr = 3;
    n_dim = 2;
end
properties (Dependent)
    n_fr
end
methods
    function W = SimEn(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        varargin2props(W, varargin);
    end
    function n_fr = get.n_fr(W)
        n_fr = W.t_max * W.refresh_rate;
    end
    function en = main(W, varargin)
        % OPTIONS:
        % 'mu', 0 % expanded to (tr, fr, dim)
        % 'conv_mean', 0.15
        % 'conv_std', 0.1
        S = varargin2S(varargin, {
            'mu', 0 % expanded to (tr, fr, dim)
            'conv_mean', 0.15
            'conv_std', 0.1
            });
                
        mu = S.mu + zeros(W.n_tr, W.n_fr, W.n_dim);        
        en = normrnd(mu, 1 / W.n_fr);
        
        conv_mean = S.conv_mean + zeros(1, 2);
        conv_std = S.conv_std + zeros(1, 2);

        t = (0:(W.n_fr - 1)) / W.refresh_rate;        
        for dim = 1:W.n_dim
            kernel = sums1(gampdf_ms(t, conv_mean(dim), conv_std(dim))); 
            en(:,:,dim) = conv_t(en(:,:,dim)', kernel(:))';
        end
    end
end
end