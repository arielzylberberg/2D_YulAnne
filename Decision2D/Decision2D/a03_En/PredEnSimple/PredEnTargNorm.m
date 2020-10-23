classdef PredEnTargNorm < PredEnSimple
methods
    function W = PredEnTargNorm(varargin)
        W.policy = 'targnorm'; % targetwise with normalization
        W.init_params0;
        if nargin > 0
            W.init(varargin{:});
        end
    end
    function init(W, varargin)
        varargin2props(W, varargin);
    end
    function init_params0(W)
        W.add_params({
%             {'x0', 0.5, 0.001, 1} % starting point for accumulated evidence
%             {'i0', 0.5, 0, 1} % constant input
%             {'a', 0.05, 0, 1} % auto-excitation
%             {'b', 0.035, 0, 0.33} % inhibition by sum of other targets
%             {'k', [0.016, 0.012], [0, 0], [50, 50]} % sensitivity to each dim
%             {'bound', 100, 0, 200} % bound height
            {'x0', 0.5, 0.001, 1} % starting point for accumulated evidence
            {'i0', 0.1, 0, 1} % 0.5, 0, 1} % constant input
            {'a', 0.01, 0, 1} % 0.05, 0, 1} % auto-excitation
            {'b', 0.005, 0, 1} %  0.035, 0, 0.33} % inhibition by sum of other targets
            {'k', 1e0 * [1.6, 0.4], [0, 0], [50, 50]} % [0.016, 0.012], [0, 0], [50, 50]} % sensitivity to each dim
            {'bound', 100, 0, 200} % bound height
            });
    end
    function [ch, rt, res, ens] = pred(W, en, bound, varargin)
        %%
        gain = bsxfun(@times, ones(size(en)), reshape2vec(W.th.k, 3));
        n_targ = 4;
        if nargin < 2 || isempty(bound) || size(bound, 3) < n_targ
            bound = ones([sizes(en, 1:2), n_targ]) * W.th.bound;
        end
        %%
        [ch, rt, res, ens] = W.pred_ch_rt(en, gain, bound, varargin{:});
    end
    function [ch, rt, res, ens] = pred_ch_rt(...
            W, en, gain, bound, varargin)
        % en(tr, fr, dim)
        % gain(tr, fr, dim)
        % bound(tr, fr)
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
            'th', W.th
            });
        th = S.th;
        
        n_dim = 2;
        n_targ = 2 .^ n_dim;
        
        n_tr = size(en, 1);
        n_fr = size(en, 2);        
        dt = 1 / W.refresh_rate;
        
        if isempty(S.noise)
            noise = normrnd(0, sqrt(S.diffusion_per_sec * dt), ...
                [n_tr, n_fr, n_targ]); % targetwise noise
        else
            noise = S.noise;
        end
        
        %% ev for each target
        ev0 = en .* reshape2vec(th.k, 3);
        ev(:,:,1) = -ev0(:,:,1) -ev0(:,:,2);
        ev(:,:,2) = +ev0(:,:,1) -ev0(:,:,2);
        ev(:,:,3) = -ev0(:,:,1) +ev0(:,:,2);
        ev(:,:,4) = +ev0(:,:,1) +ev0(:,:,2);
        
        %% simulate each time step
        cum_ev = zeros(n_tr, n_fr, 2 .^ n_dim);
        cum_ev1 = W.th.x0 + zeros(n_tr, 1, 2 .^ n_dim);
        cum_ev(:,1,:) = cum_ev1;
        
        reached_bound_already = false(n_tr, 1);
        ch = nan(n_tr, 1);
        td = nan(n_tr, 1);
        
        for it = 2:n_fr
            % Cache prev activity
            cum_ev0 = cum_ev1;
            
            % Add I0
            cum_ev1 = cum_ev1 + th.i0;
            
            % Add self-excitation
            cum_ev1 = cum_ev1 + cum_ev0 .* th.a;
            
            % Subtract inhibition
            cum_ev1 = cum_ev1 - sum(cum_ev0, 3) .* th.b ...
                + cum_ev0 .* th.b;
            
            % Add momentary evidence
            cum_ev1 = cum_ev1 + ev(:,it,:);
            
            % Add noise
            cum_ev1 = cum_ev1 + noise(:,it,:);
            
            % Rectify
            cum_ev1 = max(cum_ev1, 0);
            
            % Determine if decision is reached
            [reached_bound, which_bound] ...
                = max(cum_ev1 >= bound(:,it,:), [], 3);
            new_reached_bound = reached_bound & ~reached_bound_already;
            ch(new_reached_bound) = which_bound(new_reached_bound);
            td(new_reached_bound) = it;
            reached_bound_already = ...
                reached_bound_already | new_reached_bound;

            % Store results
            cum_ev(:,it,:) = cum_ev1;
        end
        
        %% Determine choice when undecided
        undecided = ~reached_bound_already;
        [~, ch(undecided)] = max(cum_ev1(undecided,1,:), [], 3);
        td(undecided) = n_fr;
        
        %% Add Tnd
        tnd = W.get_tnd(S.tnd_mean_sec, S.tnd_std_sec, n_tr);
        rt = td + tnd;        
        rt = max(min(rt, n_fr), 1);
        
        %% Convert choice wrt targets to dims
        ch0 = ch;
        ch = nan(n_tr, 2);
        ch(:,1) = mod(ch0 - 1, 2) + 1;        
        ch(:,2) = floor((ch0 - 1) / 2) + 1;
        
        %% Output
        res = packStruct(td, gain, bound, tnd, noise);        
        if nargout >= 4
            ens = W.get_ens(rt, en);
        end        
    end    
end
methods
    function plot_vector_field(W)
        %%
        n = 4;

        xs = 0:0.2:1;
        ss = 0:0.2:(xs(end) * n);
        [x, s] = meshgrid(xs, ss);
        
        th = W.th;
        
        i = -10;
        dx = th.i0 + i + (th.a + th.b) .* x - th.b .* s;
        ds = n .* (th.i0 + th.a .* s);
        
        quiver(x, s, dx, ds);
        axis equal
        
        %%
    end
end
end