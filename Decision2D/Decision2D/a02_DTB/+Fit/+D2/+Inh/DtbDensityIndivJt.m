classdef DtbDensityIndivJt ...
        < Fit.D2.Inh.DtbDensity ...
        & Fit.D2.Inh.DtbSigmaSq
    % Fit.D2.Inh.DtbDensityIndivJt
    %
    % Allows specifying different drift, bound, and sigmaSq for
    % each joint condition.
    %
    % Since bounds are independent of conditions, they are not modified
    % in this class.
    %
    % Here,
    % td_together : {dim}(t, cond12, ch) = p(tr, ch | cond12)
    % unabs_together : {dim}(t, y, cond12) = p(tr, t, y, | cond12)
    %
    % 2016 YK wrote the initial version.
    
%% Init
methods
    function W = DtbDensityIndivJt(varargin)
        if nargin > 0
            W.init(varargin{:});
        end
    end
end
%% Calculation
%% ----- calc_Td_together and its submethods
methods
    function sigmaSq = get_sigmaSq_together(W, dim, dim_1st)
        % sigmaSq = get_sigmaSq_together(W, dim, dim_1st)
        
        sigmaSq_fac = W.get_sigmaSq_fac_together(dim, dim_1st);
        sigmaSq = bsxfun(@times, ...
            sigmaSq_fac, W.SigmaSqs{dim}.get_sigmaSq_vec);
    end
end
%% ----- calc_Td_together_first and its submethods
methods
    function [td_together_dim_1st, td_together_any_ch] = ...
            get_td_together_dim_1st(W, td_together)
        n_dim = 2; % CONST
        
        for dim = n_dim:-1:1
            td_together_any_ch{dim} = sum(td_together{dim}, ...
                W.dim_td_together.ch);
        end
        
        %% Determine which dim is first for each joint condition
        n_conds = W.Data.get_nConds;
        
        for cond1 = n_conds(1):-1:1
            for cond2 = n_conds(2):-1:1
                cond12 = sub2ind(n_conds, cond1, cond2);
                
                p = [td_together_any_ch{1}(:, cond12), ...
                     td_together_any_ch{2}(:, cond12)]; 
                
                % which dim was first
                [~, p_1st] = min_distrib(p);
                
                % t * dim_1st * cond1 * cond2
                td_together_dim_1st(:, :, cond1, cond2) = p_1st;
            end
        end
    end
    function td_together_first = get_td_together_first(W, ...
            td_together_dim_1st, td_together, td_together_any_ch)
        % td_together_first = get_td_together_first(W, ...
        %     td_together_dim_1st, td_together, td_together_any_ch)
        
        n_dim = 2; % CONST
        n_conds = W.Data.get_nConds;
        
        for dim = n_dim:-1:1
            for cond1 = n_conds(1):-1:1
                for cond2 = n_conds(2):-1:1
                    cond12 = sub2ind(n_conds, cond1, cond2);
                    
                    for ch = 2:-1:1
                        % {dim}(t, cond1, cond2, ch)
                        td_together_first{dim}(:, cond1, cond2, ch) = ...
                            td_together_dim_1st(:, dim, cond1, cond2) ...
                            .* nan0(td_together{dim}(:, cond12, ch) ...
                                 ./ td_together_any_ch{dim}(:, cond12));
                    end                            
                end
            end            
        end
    end
    function unabs_together_first = get_unabs_together_first(W, ...
            unabs_together, td_together_first)

        n_ch = 2; % CONST
        n_dim = 2; % CONST
        
        for dim = n_dim:-1:1
            o_dim = n_dim + 1 - dim;

            n_conds = W.Data.get_nConds;        

            % sum across y should match td_together_first{o_dim}
            unabs_together_given_t = nan0(bsxfun(@rdivide, ...
                unabs_together{dim}, ...
                sum(unabs_together{dim}, W.dim_unabs_together.y)));

            for cond1 = n_conds(1):-1:1
                for cond2 = n_conds(2):-1:1
                    % Different from DtbDensity
                    cond12 = sub2ind(n_conds, cond1, cond2);

                    for o_ch = n_ch:-1:1
                        % {dim}(t, y, cond1, cond2, o_ch)
                        unabs_together_first{dim}(:, :, cond1, cond2, o_ch) ...
                            = bsxfun(@times, ...
                            ... % Different from DtbDensity
                            unabs_together_given_t(:, :, cond12), ...
                            td_together_first{o_dim}(:, cond1, cond2, o_ch));                        
                    end
                end
            end
            if W.TO_DEBUG
                try
                    % Needs to match only when sum(unabs) > 0
                    [tf, s1, s2] = assert_isequal_within( ...
                        sums(unabs_together_first{dim}, ...
                            W.dim_unabs_together_first.y, true), ...
                        td_together_first{o_dim} ...
                            .* ... 
                            (sums(unabs_together_first{dim}, ...
                                W.dim_unabs_together_first.y, true) > 0), ...
                        0.01, 'relative_tol', false);
                    assert(tf);
                catch err
                    if W.TO_DEBUG > 1
                        fig_tag('debug');
                        subplot(2,1,1); 
                        plot(s1(:, 1, 5, 1)); 
                        subplot(2,1,2); 
                        plot(s2(:, 1, 5, 1))

                        %%
                        rethrow(err);
                    else
                        warning(err_msg(err));
                    end
                end                    
            end

            % Shift one step, because unabsorbed density at t=1
            % shouldn't be absorbed at the same time step.
            % See dtb.pred.spectral_dtbAA_ns_drift_var_y0.
    %             unabs_together_first{dim} = ...
    %                 circshift(unabs_together_first{dim}, 1, ...
    %                     W.dim_unabs_together_first.t);
    %             unabs_together_first{dim}(1, :, :, :, :) = 0;
        end
    end
    function sum_td_within_cond12 = ...
            get_sum_td_within_cond12(~, sum_td_within_cond)
        % sum_td_within_cond12 = ...
        %   get_sum_td_within_cond12(W, sum_td_within_cond)
        
        sum_td_within_cond12 = ...
            sum_td_within_cond{1} .* ...
            sum_td_within_cond{2};
    end
end
%% ----- calc_Td_alone and its submethods
methods
    function drift_cond_t = get_drift_cond_t_dim(W, dim)
        % drift_cond_t = get_drift_cond_t_dim(W, dim)
        
        Drift = W.get_Drift(dim);
        drift_cond_t = Drift.get_drift_cond_t;
    end
    function sigmaSq = get_sigmaSq_alone(W, dim)
        sigmaSq = W.SigmaSqs{dim}.get_sigmaSq_vec;
    end
end
%% Drift
methods
    function set_Drift1(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.set_Drift_dim(obj_or_name, 1);
    end
    function set_Drift2(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.set_Drift_dim(obj_or_name, 2);
    end
    function set_Drift_dim(W, obj_or_name, dim)
        if nargin < 2, obj_or_name = 'Const'; end
        drift = sprintf('Drift%d', dim);
        
        W.(drift) = W.enforce_class( ...
            'Fit.D2.IrrIxn.Drift', obj_or_name);
        W.(drift).dim_rel_W = dim;
        W.set_sub_from_props({drift});
        W.(drift).customize_th_for_Data(dim);
        
        %% Update SigmaSq
        sigmaSq = sprintf('SigmaSq%d', dim);
        if ~isempty(W.(sigmaSq))
            W.(sigmaSq).set_Drift(W.(drift));
        end
    end
end
%% SigmaSq
methods
    function set_SigmaSq1(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.set_SigmaSq_dim(obj_or_name, 1);
    end
    function set_SigmaSq2(W, obj_or_name)
        if nargin < 2, obj_or_name = 'Const'; end
        W.set_SigmaSq_dim(obj_or_name, 2);
    end
    function set_SigmaSq_dim(W, obj_or_name, dim)
        if nargin < 2, obj_or_name = 'Const'; end
        sigmaSq = sprintf('SigmaSq%d', dim);
        
        W.(sigmaSq) = W.enforce_class( ...
            'Fit.D2.IrrIxn.SigmaSq', obj_or_name);
        W.(sigmaSq).dim_rel_W = dim;
        W.set_sub_from_props({sigmaSq});
        W.(sigmaSq).customize_th_for_Data(dim);
        
        %% Update Drift
        drift = sprintf('Drift%d', dim);
        W.(sigmaSq).set_Drift(W.(drift));
    end
end
end