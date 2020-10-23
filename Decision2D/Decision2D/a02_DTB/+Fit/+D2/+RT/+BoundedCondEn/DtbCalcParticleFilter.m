classdef DtbCalcParticleFilter < Fit.D2.Common.CommonWorkspace
    % Simulate absorbed and weighted unabsorbed particles.
    %
    % 2015 YK wrote the initial version.
properties
end
methods
    function [td_pdf, traj] = get_pred_td_tr_t_ch_pdf(W)
        % td_pdf(t, tr, 1, ch1, ch2) = probability
        % traj(tr, dim, t) = particle's evidence level before absorption
        
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
                
        tr_ix = (1:n_tr)';
        y = zeros(n_tr, n_dim); % FIXIT: Use y0
        traj = nan(n_tr, n_dim, nt);
        traj(:, :, 1) = y;
        
        % Different from here
        unabsorbed = ones(n_tr, n_dim);
        
        td_pdf = zeros([n_tr, (zeros(1, n_dim) + n_ch), n_t]);
        
        for it = 1:nt
            for i_dim = 2:-1:1
                n_unabsorbed = sum(unabsorbed(:, i_dim) > 0);
                dy = randn(n_unabsorbed, 1);
                dy = dy .* sigma .* sqrt(dt) ...
                    + drift .* dt;
                
                
                
                % If absorbed, give all weights to the absorbed bound.
                % Rationale: it is likely that the probability to go to 
                % the unabsorbed bound and being unabsorbed at all is
                % low anyway. That would not be worth simulating further.
                
            end
        end
    end
end
end