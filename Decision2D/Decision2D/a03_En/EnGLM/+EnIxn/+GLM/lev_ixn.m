function [est, ci, n_tr_in_fr] = lev_ixn(ens_mat, ch, cond, varargin)
% [est, ci, n_tr_in_fr] = xcorr_use(ens_mat, chs, conds)
%
% INPUT:
% ens_mat{dim}(tr, fr)
% chs(tr, dim)
% cond(tr, dim)
%
% dim: 1 is dim_rel, 2 is dim_irr
%
% OPTION:
% 'lev_kind', 'beta' % 'beta'|'bsame'|'bsameonly'
% 'smooth_fr', 0 % stdev of the gaussian kernel
%
% OUTPUT:
% est(fr)
% ci(fr, [lb, ub])

S = varargin2S(varargin, {
    'lev_kind', 'beta' % 'beta'|'bsame'|'bsameonly'
    'smooth_fr', 0 % stdev of the gaussian kernel
    });

%% Init input
n_fr = size(ens_mat{1}, 2);
n_tr_in_fr = zeros(n_fr, n_fr);

en_rel = ens_mat{1};
en_irr = ens_mat{2};
ch_rel = ch(:, 1);
ch_irr = ch(:, 2);

if S.smooth_fr > 0
    en_rel = smooth_gauss_nan(en_rel', S.smooth_fr)';
    en_irr = smooth_gauss_nan(en_irr', S.smooth_fr)';
end

%% Init output
est = zeros(n_fr, n_fr);
se = zeros(n_fr, n_fr);        

%% Regressors
roni0 = [
    bml.stat.ind_cols(cond(:,1)), ...
    bml.stat.ind_cols(cond(:,2)) ...
    ];

%% Loop across fr_rel x fr_irr
switch S.lev_kind
    case 'beta'
        en_ch_irr = bsxfun(@times, en_irr, sign(ch_irr - 0.5));

        for fr1 = 1:n_fr
            for fr2 = 1:n_fr
                en_rel_ch_irr = en_rel(:,fr1) .* en_ch_irr(:,fr2);

                x = standardize([en_rel_ch_irr, ...
                     en_rel(:, fr1), ...
                     en_irr(:, fr2), ...
                     en_ch_irr(:, fr2), ...
                     roni0]);

                res = glmwrap(x, ch_rel, 'binomial');
                est(fr1, fr2) = res.b(2);
                se(fr1, fr2) = res.se(2);
                n_tr_in_fr(fr1, fr2) = ...
                    sum(~isnan(en_rel(:, fr1)) & ~isnan(en_irr(:, fr2)));
            end
        end

    case 'bsame'
        ch_same = ch_rel == ch_irr;

        for fr1 = 1:n_fr
            for fr2 = 1:n_fr
                en_rel_irr = en_rel(:,fr1) .* en_irr(:,fr2);

                x = standardize([en_rel_irr, ...
                     en_rel(:, fr1), ...
                     en_irr(:, fr2), ...
                     roni0]);

                res = glmwrap(x, ch_same, 'binomial');
                est(fr1, fr2) = res.b(2);
                se(fr1, fr2) = res.se(2);
                n_tr_in_fr(fr1, fr2) = ...
                    sum(~isnan(en_rel(:, fr1)) & ~isnan(en_irr(:, fr2)));
            end
        end    

    case 'bsameonly'
        ch_same = ch_rel == ch_irr;
        for fr1 = 1:n_fr
            for fr2 = 1:n_fr
                en_rel_irr = en_rel(:,fr1) .* en_irr(:,fr2);

                x = standardize([en_rel_irr, ...
                     roni0]);

                res = glmwrap(x, ch_same, 'binomial');
                est(fr1, fr2) = res.b(2);
                se(fr1, fr2) = res.se(2);
                n_tr_in_fr(fr1, fr2) = ...
                    sum(~isnan(en_rel(:, fr1)) & ~isnan(en_irr(:, fr2)));
            end
        end                
end

ci = cat(3, est - se, est + se);        
end