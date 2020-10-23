classdef Ev2PchD2 ...
        < IxnKernel.EvTime.CommonWorkspaceD2
    % IxnKernel.RevCor.D2.Ev2PchD2 - Converts evidence to Pch
    
%% Settings
properties
    EPs = {IxnKernel.RevCor.D1.Ev2PchD1, IxnKernel.RevCor.D1.Ev2PchD1};
end
%% Placeholder. May define if eneded.
properties
    ev
    ch
    n_trial
end
%% Results
properties
    % res_beta(dim)
    % .rt_incl_ms
    % .align
    % .itv_len_ms
    % .t_st_ms(1,t)
    % .slope(1,t)
    % .bias(1,t)
    res_beta = [];
end
%% Init
methods
    function EP = Ev2PchD2(varargin)
        EP.props_to_share_Ev = {'EPs'};
        
        if nargin > 0
            EP.init(varargin{:});
        end
    end
    function init(EP, varargin)
        varargin2props(EP, varargin, true);
    end
    function main(EP)
        EP.batch_fit_intervals;
        EP.plot_and_save_all;
    end
end
%% Fit
methods
    function batch_fit_intervals(EP, varargin)
        S_batch = varargin2S(varargin, {
            'align', {'rt', 'st'}
            });
        [Ss, n] = factorizeC(S_batch);
        for ii = 1:n
            S = Ss(ii);
            C = S2C(S);
            
            EP.fit_intervals(C{:});
        end
    end
    function res_beta = fit_intervals(EP, varargin)
        for dim = EP.n_dim:-1:1
            res_beta(dim) = EP.EPs{dim}.fit_intervals(varargin{:});
        end
        if nargout == 0
            EP.res_beta = res_beta;
        end
    end
end
%% f_p_ch_given_util
methods
    function f = get_f_p_ch_given_util(EP)
        assert(~isempty(EP.res_beta));
        f = @(ev, itv) EP.f_p_ch_given_util(ev, itv, ...
            EP.res_beta);
    end
    function p_ch_given_util = f_p_ch_given_util(EP, ev, n_used, res_beta)
        % p_ch_given_util = f_p_ch_given_util(EP, ev, n_used, row)
        % 
        % p_ch_given_util(tr,itv,dim) 
        % = Pr(ch(tr,dim)==1 | ev(tr,itv,dim) is used)
        %
        % n_used : 1 x n_dim vector
        % p_ch_given_util(tr, itv, used1, used2)
        % ev(tr, 1, dim)
        
        for dim = EP.n_dim:-1:1
            p_ch_given_util(:,:,dim) = EP.EPs{dim}.f_p_ch_given_util( ...
                ev(:,:,dim), ...
                n_used(dim), ...
                res_beta(dim));
        end
        
%         p_ch_given_util(:,:,2,2) = p_ch_dim(:,:,1) .* p_ch_dim(:,:,2);
%         p_ch_given_util(:,:,1,2) = (1 - p_ch_dim(:,:,1)) .* p_ch_dim(:,:,2);
%         p_ch_given_util(:,:,2,1) = p_ch_dim(:,:,1) .* (1 - p_ch_dim(:,:,2));
%         p_ch_given_util(:,:,1,1) = (1 - p_ch_dim(:,:,1)) .* p_ch_dim(:,:,2);
    end
end
end