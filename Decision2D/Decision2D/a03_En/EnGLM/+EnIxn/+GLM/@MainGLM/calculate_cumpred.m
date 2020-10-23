function [est, ci] = calculate_cumpred(W, en_rel, ch_rel, varargin)
    S = varargin2S(varargin, {
        'dt', 0.08
        't_max', 2.4 % Long enough
        'to_bootstrap', false
        'n_boot', 10
        'ci_prctile', 100 * normcdf([-1, 1])
        'n_fold', 10
        });

    %% Subsample
    dt_ratio = round(S.dt / W.dt);
    en_rel1 = standardize(subsample(en_rel, dt_ratio, 2));
    
    %% Lassoglm - incremental
    nt = round(S.t_max / S.dt); % 30; % nt;
    n_tr = size(en_rel1, 1);

    offset0 = zeros(n_tr, 1);
    r = zeros(nt, 1); % coefficient of discrimination
    offset = zeros(n_tr, nt);
    
    [~,~,dcond0] = unique(W.Data.ds.cond, 'rows');
        
    %%
    fprintf('Starting calculate_cumpred for %d timepoints: ', nt);
    for ii = 1:nt
        
        x = en_rel1(:, ii);
        tr_incl = ~any(isnan(x), 2) & ~isnan(ch_rel);
        n_tr1 = nnz(tr_incl);
        
        if n_tr1 < S.n_fold * 5
            r(ii) = nan;
            continue;
        end
        
        dcond = dcond0(tr_incl, :);
        x = x(tr_incl);
        ch_rel1 = ch_rel(tr_incl);
        offset1 = offset0(tr_incl);
        
        C = varargin2C({
            'CV', cvpartition(dcond, 'KFold', S.n_fold)
            'Alpha', 0.001
            });
        C_glmval = {};
        if ii > 1
            C = varargin2C(C, {
                'Offset', offset1
                });
            C_glmval = varargin2C(C_glmval, {
                'offset', offset1
                });
        end
                
        [b0, info] = lassoglm(x, ch_rel1, 'binomial', C{:});
        ix = info.IndexMinDeviance;
        b = [
            info.Intercept(ix)
            b0(:, ix)
            ];
        
        %%
        x(isnan(x)) = 0;
        yhat = glmval(b, x, 'logit', C_glmval{:});
        r(ii) = bml.stat.coef_disc(yhat, ch_rel1);
        
        offset1 = [ones(n_tr1, 1), x] * b + offset1;
        offset0(tr_incl) = offset1;
        offset(:, ii) = offset0;
        
        fprintf('%d ', ii);
        if mod(ii, 10) == 0
            fprintf('\n'); 
        end
    end
    fprintf('Done.\n');
    
    %%
    t = (1:nt) * S.dt - S.dt;
%     plot(t, r(1:nt), 'o-');
%     xlabel('Time included up to (s)');
%     ylabel('Coefficient of Discrimination');
%     bml.plot.beautify;
    
    %% Interpolate back
    est = interp1(t(:), r(:), W.t(:));
    
    
    %% Get CI
    if S.to_bootstrap
        nt0 = numel(W.t);
        est_boot = zeros(nt0, S.n_boot);
        C = varargin2C({
            'to_bootstrap', false
            }, S);
        
        est_boot(:,1) = est;
        
        fprintf('Starting %d-1 bootstraps of calculate_cumpred\n', ...
            S.n_boot);
        parfor i_boot = 2:S.n_boot % parfor
            % Sample with replacement within each condition
            ix_boot = bml.stat.randsample_group(dcond0);
            
            est1 = W.calculate_cumpred( ...
                en_rel(ix_boot, :), ch_rel(ix_boot, :), C{:});
            
            disp(size(est1));
            est_boot(:, i_boot) = est1;
            fprintf('%d/%d bootstrap done.\n', i_boot, S.n_boot);
        end
        est = nanmedian(est_boot, 2);
        ci = prctile(est_boot, S.ci_prctile, 2);
    else
        ci = zeros(size(est));
    end
    
    %% Lassoglm - non-incremental
%     t_max = 30; % nt;
%     
%     for ii = 16:t_max % 1:15 % 5:5:15 % 1:nt
%         %%
%         t_incl = 1:ii;
%         x = en_rel1(:, t_incl);
%         
%         [b0, info] = lassoglm(x, ch_rel1, 'binomial', ...
%             'CV', 20, ...
%             'Alpha', 0.001);
% 
%         %%
% %         lassoPlot(b, info, 'plottype', 'CV');
% 
%         ix = info.IndexMinDeviance;
%         b = [
%             info.Intercept(ix)
%             b0(:, ix)
%             ];
%         
%         %%
%         x(isnan(x)) = 0;
%         yhat = glmval(b, x, 'logit');
%         r(ii) = bml.stat.coef_disc(yhat, ch_rel1);
%     end
%     
%     %%
%     t = (1:t_max) * S.dt - S.dt;
%     plot(t, r(1:t_max));
%     xlabel('Time included up to (s)');
%     ylabel('Coefficient of Discrimination');
%     bml.plot.beautify;
    
    %%
    
end