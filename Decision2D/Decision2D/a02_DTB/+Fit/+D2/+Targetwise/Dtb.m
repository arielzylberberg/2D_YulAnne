classdef Dtb < Fit.D2.Common.CommonWorkspace
    % Fit.D2.Targetwise.Dtb
    %
    % 2015 YK wrote the initial version.
properties (SetAccess = protected)
    Drift % Gives 2x2 drifts
    Bound % Gives 2x2 bounds
end
methods
    function W = Dtb
        W.add_deep_copy({'Drift', 'Bound'});
        % Remove parameters within component Dtbs.
        
        W.set_Data;
        W.set_Drift;
        W.set_Bound;
    end
    function pred(W)
        W.Data.set_Td_pred_pdf(W.get_Td_pred_pdf);
    end
    function Td_pred_pdf = get_Td_pred_pdf(W)
        bound_t_ch = W.Bound.get_bound_t_ch;
        assert(isequal(size(bound_t_ch), ...
            W.get_required_size_bound_t_ch));
        
        size_pdf = W.Data.get_size_RT_Td_pdf;
        td_raw = zeros(size_pdf);
        
        % One-bounded diffusion
        for ch1 = 1:2
            for ch2 = 1:2
                drift_cond_t = W.Drift.get_drift_cond_t(ch1, ch2);
                assert(isequal(size(drift_cond_t), ...
                    W.get_required_size_drift_cond_t));
                
%                 % Only constant bound is allowed for now
%                 assert(all(vVec(diff(bound_t_ch,[],1) == 0))); 
%                 
                % Only constant drift is allowed for now
                assert(all(vVec(diff(drift_cond_t,[],3) == 0))); 

                t = W.Data.t;
                drift = hVec(drift_cond_t(:,:,1));
                bound = bound_t_ch(ch1, ch2, 1);
        
                D = dtb.pred.analytic_dt1b(drift, t, bound);
        
                size_within_choice = size_pdf(1:3);
                td_raw(:,:,:,ch1,ch2) = reshape(D.up.pdf_t, ...
                    size_within_choice);
            end
        end
        td_raw = bsxfun(@rdivide, td_raw, sums(td_raw, [1, 4, 5]));
        ch_raw = sum(td_raw, 1);
        
        % Choose the first absorbed
        n_cond1 = size(td_raw, 2);
        n_cond2 = size(td_raw, 3);
        for cond1 = n_cond1:-1:1
            for cond2 = n_cond2:-1:1
                % td_min{L}(t) = P(min(tCh1L, tCh2L) = t | cond1, cond2)
                % td_first_ch1{L}(t,K) = P(t = tChKL < tChML | cond1, cond2)
                % where
                %   K and M refers to values of ch1,
                %   M = 3 - K, and
                %   L = ch2.
                for ch2 = 2:-1:1
                    [td_min{ch2}, td_first_ch1{ch2}] = ...
                        bml.stat.min_distrib( ...
                            squeeze(td_raw(:, cond1, cond2, :, ch2)), ...
                            'sum');
                end
                
                % td_min(t) = P(min(td_min{1}, td_min{2}) = t | cond1, cond2)
                % td_first(t,L) = P(t = td_min{L} < td_min{L} | cond1, cond2)
                % where L = ch2.
                [~, td_first_ch2] = ...
                    bml.stat.min_distrib([td_min{1}, td_min{2}], 'sum');
                
                % td_first_ch4(t,K,L)
                % = P(t = tChKL < min(tChML, tChKQ, tChMQ) | cond1, cond2)
                % = P(t = tChKL < tChML 
                %     | t = min(tChKL, tChML) < min(tChKQ, tChMQ)
                %     | cond1, cond2)
                % where
                %   K and M refers to values of ch1,
                %   M = 3 - K.
                %   L = ch2, and
                %   Q = 3 - L.
                for ch2 = 2:-1:1
                    td_first_ch4(:,:,ch2,cond1,cond2) = ...
                        bsxfun(@times, ...
                            td_first_ch2(:,ch2), ...
                            bml.math.nan0(bsxfun(@rdivide, ...
                                td_first_ch1{ch2}, ...
                                sum(td_first_ch1{ch2}, 2))));
                end
            end
        end
        
        Td_pred_pdf = permute(td_first_ch4, [1,4,5,2,3]);
    end
    %% Check
    function siz = get_required_size_drift_cond_t(W)
        siz = [W.Data.get_nConds, W.Data.nt];
    end    
    function siz = get_required_size_bound_t_ch(W)
        siz = [W.Data.nt, 2, 2];
    end
    %% Get/Set
    function set_Drift(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Drift = W.enforce_class('Fit.D2.Targetwise.Drift', obj_or_name);
        W.set_sub_from_props({'Drift'});
    end
    function set_Bound(W, obj_or_name)
        if nargin < 2, obj_or_name = ''; end
        W.Bound = W.enforce_class('Fit.D2.Targetwise.Bound', obj_or_name);
        W.set_sub_from_props({'Bound'});
    end
end
end