function [td_pdf, unabs, S] = calc_dtb(drift_cond_t, t, bound_t_ch, y, varargin)
    % [td_pdf, unabs, S] = calc_dtb(drift_cond_t, t, bound_t_ch, y, varargin)
    %
    % INPUT
    % -----
    % drift_cond_t: cond * t
    % t : a vector
    % bound_t_ch: t * ch
    % y : a vector
    %
    % OPTIONS
    % -------
    % 'y0', 0 % Gives p0 if necessary
    % 'p0', []
    % 'ny', length(y)
    % 'calc_unabs', nargout >= 2
    % 'sigmaSq', 1
    % ...
    % 'is_constant_drift', []
    % 'is_constant_bound', []
    % 'is_constant_p0', []
    % ...
    % 'apply_sign_rule', false
    % 't_sign_rule_sec', [] % Defaults to t(end)
    %
    % OUTPUT
    % ------
    % td_pdf: t * cond * ch
    % unabs: t * y * cond

    %% Inputs
    S = varargin2S(varargin, {
        'y0', 0 % Convert to a delta function if necessary
        'p0', []
        'ny', length(y)
        'calc_unabs', nargout >= 2
        'sigmaSq', 1
        'sigmaSq_st', 0
        ...
        'is_constant_drift', []
        'is_constant_bound', []
        'is_constant_p0', []
        'is_delta_p0', []
        ...
        'to_interp_drift', false
        ...
        'apply_sign_rule', false % true % false
        't_sign_rule_sec', [] % Defaults to t(end)
        ...
        'dtb_fun_used', ''
        ...
        'min_n_cond_to_use_parallel', 200
        });

    if S.apply_sign_rule
        S.calc_unabs = true;
    end

    dim_time = 2;
    if isempty(S.is_constant_drift)
        S.is_constant_drift = ...
            all(vVec( ...
                diff(drift_cond_t, [], dim_time) == 0 ...
            ));
    end
    if isempty(S.is_constant_bound)
        S.is_constant_bound = ...
            all(bound_t_ch(:,1) == bound_t_ch(1,1)) && ...
            all(bound_t_ch(:,2) == bound_t_ch(1,2));
    end
    if isempty(S.is_constant_p0)
        S.is_constant_p0 = isempty(S.p0) || isvector(S.p0);
    end
    if isempty(S.p0)
        S.p0 = delta_fun(S.y0, y);
    end

    if S.sigmaSq_st > Fit.D1.Bounded.Dtb.MIN_SIGMASQ_ST
        st_filt = normpdf(y, 0, sqrt(S.sigmaSq_st));
        st_filt = st_filt ./ sum(st_filt);

        sum_p0 = sum(S.p0);
        S.p0 = conv(S.p0, st_filt, 'same');
        S.p0 = bsxfun(@times, ...
                bsxfun(@rdivide, S.p0, sum(S.p0)), sum_p0);

        S.is_delta_p0 = false;
    else
        S.is_delta_p0 = true;
    end
    
    S = copyFields(S, packStruct(drift_cond_t, t, bound_t_ch, y));
    
    %%
    drift_vec = drift_cond_t(:,1);
    n_drift = length(drift_vec);
    
    %% Parallelization
    global to_use_parallel
    if n_drift >= S.min_n_cond_to_use_parallel && ~is_in_parallel
        assert(S.is_constant_p0, 'non-constant p0 is not supported yet!');
        
        if isempty(to_use_parallel)
            to_use_parallel = true;
        end
        if to_use_parallel
            pool = gcp;
            n_worker = pool.NumWorkers;
        else
            n_worker = 1;
        end
        ix_drift = ceil((1:n_drift) / n_drift * n_worker);
        
        drift_cond_ts = cell(n_worker, 1);

        n_argout = max(1, nargout);
        outs = repmat({cell(1, n_argout)}, [n_worker, 1]);
        
        % Package into settings
        for i_worker = n_worker:-1:1
            S1 = S;
            
            % Prevent infinite recursion
            S1.min_n_cond_to_use_parallel = inf; 
            
            % Copy settings
            S_par(i_worker) = S1;
            
            % Ration drifts
            incl = ix_drift == i_worker;
            drift_cond_ts{i_worker} = drift_cond_t(incl, :);
            
            % Ration sigmaSq
            if size(S1.sigmaSq, 1) ~= 1
                S_par(i_worker).sigmaSq = S1.sigmaSq(incl, :);
            end
        end
        
        % DEBUG
        if to_use_parallel
            parfor i_worker = 1:n_worker
    %         for i_worker = 1:n_worker
                C = varargin2C(S_par(i_worker));
                [outs{i_worker}{1:n_argout}] = dtb.pred.calc_dtb( ...
                    drift_cond_ts{i_worker}, ...
                    t, bound_t_ch, y, C{:});
            end
        else
            for i_worker = 1:n_worker
                C = varargin2C(S_par(i_worker));
                [outs{i_worker}{1:n_argout}] = dtb.pred.calc_dtb( ...
                    drift_cond_ts{i_worker}, ...
                    t, bound_t_ch, y, C{:});
            end
        end
        
        outs = cat(1, outs{:});
        if n_argout >= 1
            td_pdf = cat(2, outs{:,1});
        end
        if n_argout >= 2
            unabs = cat(3, outs{:,2}); 
        end
        
        return;
    end

    %% Process
    if S.to_interp_drift
        assert(S.is_constant_drift);
        assert(S.is_constant_p0);
        if all(S.sigmaSq == 1)
            S.sigmaSq = 1;
        end
        assert(isequal(S.sigmaSq, 1));

        D = dtb.pred.dtb_interp_drift( ...
            drift_vec, t, ...
            bound_t_ch(:,2), bound_t_ch(:,1), ...
            y, S.p0, S.calc_unabs);
        if S.calc_unabs
            D.notabs.pdf = permute(D.notabs.pdf, [1 3 2]);
        end

        S.dtb_fun_used = 'dtb.pred.dtb_interp_drift';

    elseif S.is_constant_drift && S.is_constant_bound ...
            && S.is_constant_p0
        % Analytic
        if S.calc_unabs
            C = {S.ny};
        else
            C = {};
        end

        % SigmaSq
        if all(S.sigmaSq(:) == 1) && S.is_delta_p0
            D = dtb.pred.analytic_dtb( ...
                drift_vec', t, ...
                bound_t_ch(1,2), bound_t_ch(1,1), ...
                0, C{:});

            S.dtb_fun_used = 'dtb.pred.analytic_dtb';

        elseif all(S.sigmaSq(:) == 0)
            if all(drift_vec(:) == 0)
                % Just use one convention. No need to implement two 
                % different interfaces.
                D = dtb.pred.no_drift_no_diffusion( ...
                        n_drift, t, bound_t_ch(:,2), bound_t_ch(:,1), ...
                        y, S.p0, S.calc_unabs);
%                     D = dtb.pred.no_drift_no_diffusion( ...
%                         t, bound_t_ch(1,2), bound_t_ch(1,1), 0, C{:});
                if S.calc_unabs
                    D.notabs.pdf = permute(D.notabs.pdf, [1 3 2]);
                end

                S.dtb_fun_used = 'dtb.pred.no_drift_no_diffusion';
            else
                error('dtb.pred.drift_wo_diffusion not implemented yet!');
%                         D = dtb.pred.drift_wo_diffusion(); % TODO
            end
        elseif isscalar(S.sigmaSq)
            D = dtb.pred.analytic_dtb_sig( ...
                drift_vec', S.sigmaSq, t, ...
                bound_t_ch(1,2), bound_t_ch(1,1), ...
                0, C{:});

            S.dtb_fun_used = 'dtb.pred.analytic_dtb_sig';
        else
            % TODO: replace with analytic_dtb
            D = dtb.pred.spectral_dtbAA_sigs_ns_drift_var_y0( ...
                    drift_cond_t, t, ...
                    bound_t_ch(:,2), bound_t_ch(:,1), ...
                    y, S.p0, S.calc_unabs, S.sigmaSq);
                if S.calc_unabs
                    D.notabs.pdf = permute(D.notabs.pdf, [1 3 2]);
                end                    
%                 D = dtb.pred.analytic_dtb_sigs( ...
%                     drift_vec', S.sigmaSq, t, ...
%                     bound_t_ch(1,2), bound_t_ch(1,1), ...
%                     0, C{:});

            S.dtb_fun_used = ...
                'dtb.pred.spectral_dtbAA_sigs_ns_drift_var_y0';
        end
    else
        % Constant p0 and sigmaSq
        if S.is_constant_p0 && ...
                (isscalar(S.sigmaSq) ...
                || all(S.sigmaSq(:) == S.sigmaSq(1)))
            if all(S.sigmaSq(:) == 1)
                % All sigmaSq == 1
                if S.is_constant_drift
                    % Constant drift_cond_t
                    D = dtb.pred.spectral_dtbAA( ...
                        drift_vec, t, ...
                        bound_t_ch(:,2), bound_t_ch(:,1), ...
                        y, S.p0, S.calc_unabs);
                    if S.calc_unabs
                        D.notabs.pdf = permute(D.notabs.pdf, [1 3 2]);
                    end

                    S.dtb_fun_used = 'dtb.pred.spectral_dtbAA';
                else
                    % Changing drift_cond_t
                    D = dtb.pred.spectral_dtbAA_ns( ...
                        drift_cond_t, t, ...
                        bound_t_ch(:,2), bound_t_ch(:,1), ...
                        y, S.p0, S.calc_unabs);
                    if S.calc_unabs
                        D.notabs.pdf = permute(D.notabs.pdf, [1 3 2]);                        
                    end

                    S.dtb_fun_used = 'dtb.pred.spectral_dtbAA_ns';
                end
            elseif all(S.sigmaSq(:) == 0)
                % All sigmaSq == 0
                if all(drift_cond_t(:) == 0)
                    D = dtb.pred.no_drift_no_diffusion( ...
                        n_drift, t, bound_t_ch(:,2), bound_t_ch(:,1), ...
                        y, S.p0, S.calc_unabs);
                    if S.calc_unabs
                        D.notabs.pdf = permute(D.notabs.pdf, [1 3 2]);
                    end

                    S.dtb_fun_used = 'dtb.pred.no_drift_no_diffusion';
                else
                    error('dtb.pred.drift_wo_diffusion not implemented yet!');
%                         D = dtb.pred.drift_wo_diffusion(); % TODO
                end
            else
                % sigmaSq not all 0 or 1
                if S.is_constant_drift
                    % Constant drift_cond_t and sigmaSq
                    D = dtb.pred.spectral_dtbAA_sigmaSq( ...
                        drift_vec, t, ...
                        bound_t_ch(:,2), bound_t_ch(:,1), ...
                        y, S.p0, S.sigmaSq(1), S.calc_unabs);
                    if S.calc_unabs
                        D.notabs.pdf = permute(D.notabs.pdf, [1 3 2]);
                    end

                    S.dtb_fun_used = 'dtb.pred.spectral_dtbAA_sigmaSq';
                else
                    D = dtb.pred.spectral_dtbAA_sigs_ns_drift_var_y0( ...
                        drift_vec, t, ...
                        bound_t_ch(:,2), bound_t_ch(:,1), ...
                        y, S.p0, S.calc_unabs, S.sigmaSq);
                    
                    if S.calc_unabs
                        D.notabs.pdf = permute(D.notabs.pdf, [1 3 2]);
                    end

                    S.dtb_fun_used = 'dtb.pred.spectral_dtbAA_sigs_ns_drift_var_y0';
%                     error('Not adapted here but there may be something in dtb.pred!');
                end
            end
        else
            if all(S.sigmaSq(:) == 1)
                D = dtb.pred.spectral_dtbAA_ns_drift_var_y0( ...
                    drift_cond_t, t, ...
                    bound_t_ch(:,2), bound_t_ch(:,1), ...
                    y, S.p0, S.calc_unabs, S.sigmaSq);
                if S.calc_unabs
                    D.notabs.pdf = permute(D.notabs.pdf, [1 3 2]);
                end

                S.dtb_fun_used = ...
                    'dtb.pred.spectral_dtbAA_ns_drift_var_y0';

            elseif all(S.sigmaSq(:) == 0)
                error(['no_diffusion is not implemented for ' ...
                       'nonstationary p0 yet!']);
            else
%                 elseif min(S.sigmaSq(:)) < 1
                %% Preserves density and faster!
                D = dtb.pred.conv_dtbAA_sigs_ns_drift_var_y0( ...
                    drift_cond_t, t, ...
                    bound_t_ch(:,2), bound_t_ch(:,1), ...
                    y, S.p0, S.calc_unabs, S.sigmaSq);
                if S.calc_unabs
                    D.notabs.pdf = permute(D.notabs.pdf, [1 3 2]);
                end                  

                S.dtb_fun_used = ...
                    'dtb.pred.conv_dtbAA_sigs_ns_drift_var_y0';

%                 else
%                     %%
%                     D = dtb.pred.spectral_dtbAA_sigs_ns_drift_var_y0( ...
%                         drift_cond_t, t, ...
%                         bound_t_ch(:,2), bound_t_ch(:,1), ...
%                         y, S.p0, S.calc_unabs, S.sigmaSq);
%                     if S.calc_unabs
%                         D.notabs.pdf = permute(D.notabs.pdf, [1 3 2]);
%                     end                    
            end
        end
    end

    %% td_pdf: t * cond * ch
    td_pdf = cat(3, D.lo.pdf_t, D.up.pdf_t);

    %% unabs: t * y * cond
    if S.calc_unabs
        % t * y * cond <- cond * t * y
%             disp(size(D.notabs.pdf)); % DEBUG
        unabs = permute(D.notabs.pdf, [2, 3, 1]); 
    else
        unabs = [];
    end

    %% Sign rule
    if S.apply_sign_rule
        if isempty(S.t_sign_rule_sec)
            S.t_sign_rule_sec = t(end);
        end

%             disp(max(abs(sums(td_pdf, [1, 3]) - 1))); % DEBUG

        [td_pdf, unabs] = dtb.pred.apply_sign_rule(td_pdf, unabs, y, ...
            't_sec', t, 't_en_sec', S.t_sign_rule_sec);

%             disp(max(abs(sums(td_pdf, [1, 3]) - 1))); % DEBUG
    end
end
