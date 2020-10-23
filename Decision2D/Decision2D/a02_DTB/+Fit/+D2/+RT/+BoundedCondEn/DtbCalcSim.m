classdef DtbCalcSim < Fit.D2.RT.BoundedCondEn.DtbCalc
    %
    % 2015 YK wrote the initial version.
methods
    function W = DtbCalcSim(varargin)
        % W = DtbCalcSim(n_tr, drift, bound, tnd_st, ...)
        %
        % n_tr: scalar
        % drift: tr x t x dim
        % bound: t x ch x dim => expanded to tr x t x ch x dim
        % tnd_st: t x dim => expanded to tr x dim
        % 
        % Optional arguments:
        % sigma_alone = [1 1]; % No reason to change
        % sigmaSq_fac_bef_start = [0 0];
        % sigmaSq_fac_together  = [1 1];
        % drift_fac_together  = [1 0];
        % n_dim = 2;
        % n_ch = 2;
        % use_gpu = false;
        
        if nargin > 0
            W.init(varargin{:});
        end
    end
end
%% Calculation
methods
    function [td, ch, traj] = get_pred_td_tr_t_ch(W)
        % td: tr x 1
        % ch: tr x dim
        % traj: tr x dim x t

        n_tr = W.get_n_tr;
        n_dim = W.get_n_dim;
        n_ch = W.get_n_ch;
        
        nt = W.get_nt;
        dt = W.get_dt;
        
        drift = W.get_drift;
        bound = W.get_bound;
        tnd_st = W.get_tnd_st;
        
        drift_fac_together = W.get_drift_fac_together;
        sigmaSq_fac_together = W.get_sigmaSq_fac_together;
        sigmaSq_fac_bef_start = W.get_sigmaSq_fac_bef_start;
        
        td = repmat({nan(n_tr, n_dim)}, [1, n_ch]); % {tr x dim} x ch
        
        y = zeros(n_tr, n_dim);
        traj = zeros(n_tr, n_dim, nt);
        traj(:, :, 1) = y;
        
        absorbed = zeros(n_tr, n_dim);
        n_unabs = sum(absorbed, 1);
        
%         dy_all = randn(n_tr, nt, n_dim);
        
        for it = 2:nt
            n_dim_absorbed = sum(absorbed, 2);
            together = n_dim_absorbed == 0;
            alone = n_dim_absorbed == 1;
            
            for i_dim = n_dim:-1:1
                unabsorbed = ~absorbed(:, i_dim);
                
                % Deal with cases that already has been absorbed in the data
                % = with NaN drifts.
                absorbed_in_data = isnan(drift(:, it, i_dim)) & unabsorbed;
                
                % Tds for those trials remain NaNs so that it can be penalized.
                absorbed(:, i_dim) = absorbed(:, i_dim) | absorbed_in_data;
                unabsorbed = unabsorbed & ~absorbed_in_data;
                
                % Others
                n_unabs(i_dim) = nnz(unabsorbed);
                if n_unabs(i_dim) == 0, continue; end
                
%                 dy = dy_all(:, it, i_dim); % randn(n_tr, 1); % randn(n_unabs(i_dim), 1); % Faster but should consider conditionalization
                dy = randn(n_tr, 1); % randn(n_unabs(i_dim), 1); % Faster but should consider conditionalization
                
                c_within_tnd_st = unabsorbed & (it <= tnd_st(:, i_dim));
                c_together = unabsorbed & together & (~c_within_tnd_st);
                c_alone = unabsorbed & alone & (~c_within_tnd_st);
                
                % No drift within tnd_st by definition.
                y(c_within_tnd_st, i_dim) = y(c_within_tnd_st, i_dim) ...
                    + dy(c_within_tnd_st) ...
                        .* (sqrt(dt) ...
                        .* sigmaSq_fac_bef_start(c_within_tnd_st, i_dim));

                % DEBUG
                if any(isnan(y(:)))
                    keyboard;
                end
                
                % When together.
                y(c_together, i_dim) = y(c_together, i_dim) ...
                    + dy(c_together) ...
                        .* sqrt(dt) ...
                        .* sigmaSq_fac_together(c_together, i_dim) ...
                    + drift(c_together, it, i_dim) ...
                        .* dt ...
                        .* drift_fac_together(c_together, i_dim);

                % DEBUG
                if any(isnan(y(:)))
                    keyboard;
                end
                
                % When alone, drift and sqrt factors are always 1.
                y(c_alone, i_dim) = y(c_alone, i_dim) ...
                    + dy(c_alone) ...
                        .* sqrt(dt) ...
                    + drift(c_alone, it, i_dim) ...
                        .* dt;

                is_lo(:, i_dim) = y(:, i_dim) <= bound(:, it, 1, i_dim);
                is_up(:, i_dim) = y(:, i_dim) >= bound(:, it, 2, i_dim);

                % DEBUG
                if any(isnan(y(:)))
                    keyboard;
                end
            end
            % Process absorbed after both dim is processed.
            td{1}(is_lo & ~absorbed) = it;
            td{2}(is_up & ~absorbed) = it;
            absorbed = absorbed | is_lo | is_up; 
            traj(:, :, it) = y; % tr x dim x t            
        end
        
%         % Sign rule for the unabsorbed 
%         % - unnecessary, since observed RTs are already passed.
%         for i_dim = 2:-1:1
%             unabsorbed = ~absorbed(:, i_dim);
%             
%             is_lo(unabsorbed, i_dim) = y(unabsorbed, i_dim) < 0;
%             is_up(unabsorbed, i_dim) = y(unabsorbed, i_dim) >= 0;
%         end
%         td{1}(is_lo & ~absorbed) = it;
%         td{2}(is_up & ~absorbed) = it;
%         absorbed = absorbed | is_lo | is_up; 
%         assert(all(all(absorbed)));
        
        % Summarize td_raw
        td_raw = cat(3, td{:}); % tr x dim x ch
        for i_dim = 2:-1:1
            for i_ch = 2:-1:1
                c_ch = ~isnan(td_raw(:, i_dim, i_ch));
                ch(c_ch, i_dim) = i_ch;
            end
        end
        td = max(max(td_raw, [], 2), [], 3); % tr
        % td_pdf = []; % tr x t x ch1 x ch2 
        % td_pdf: => Not here. Only when gathering across multiple repetitions.
    end
end
%% Demo
methods (Static)
    function W = demo(kind)
        if nargin < 1, kind = 'simple'; end
        
        %%
        W = Fit.D2.RT.BoundedCondEn.DtbCalcSim.(['demo_get_W_' kind]);
        [td, ch, traj] = W.demo_plot;
    end
    function W = demo_get_W_simple
        %%
        n_tr = 2;
        
        Time = TimeAxis.TimeInheritable([], {'dt', 1, 'max_t', 2});
        nt = Time.get_nt;
        
        n_ch = 2;
        n_dim = 2;
        
        drift = ones(n_tr, nt, n_dim);
        bound = bsxfun(@times, ones(nt, n_ch, n_dim), cat(2, -1, 1));
        tnd_st = ones(nt, n_dim);
        
        W = Fit.D2.RT.BoundedCondEn.DtbCalcSim(n_tr, drift, bound, tnd_st, ...
            'Time', Time);
    end
    function W = demo_get_W_complex
        %%
        n_tr = 10;
        
        Time = TimeAxis.TimeInheritable([], {'dt', 0.1, 'max_t', 2});
        nt = Time.get_nt;
        t = Time.get_t;
        
        n_ch = 2;
        n_dim = 2;
        
        drift = bsxfun(@times, ones(n_tr, nt, n_dim), cat(3, 1, -1));
        bound = bsxfun(@times, ones(nt, n_ch, n_dim), cat(2, -1, 1));
        tnd_st = repmat(gampdf_ms(t(:), 0.1, 0.05), [1, 2]);
        
        W = Fit.D2.RT.BoundedCondEn.DtbCalcSim(n_tr, drift, bound, tnd_st, ...
            'Time', Time);
    end
end
methods
    function [td, ch, traj] = demo_plot(W)
        [td, ch, traj] = W.get_pred_td_tr_t_ch;
        % td: tr x 1
        % ch: tr x dim
        % traj: tr x dim x t
        
        nr = 3;
        nc = 1;
        t = W.get_t;
        nt = W.get_nt;
        
        tr = ':';
        subplot(nr, nc, 1);
        for i_dim = 1:2
            c_traj = squeeze(traj(tr, i_dim, :))';
            plot(t, c_traj);
            hold on;
        end
        hold off;
        grid on;
        
        %%
        for i_dim = 1:2
            subplot(nr, nc, 1 + i_dim);
            for i_ch = 1:2
                incl = ch(:, i_dim) == i_ch;
                c_td = hist(td(incl), 1:nt);
                plot(t, c_td);
                hold on;
            end
            hold off;
            grid on;
        end
    end
end
end