function ev_itv = summarize_ev(ev, varargin)
% ev_itv = summarize_ev(ev, varargin)
%
% OPTIONS:
% 'dt', 1/75 % in sec
% 'n_dt_in_itv', 15

S = varargin2S(varargin, {
    'dt', 1/75 % in sec
    'n_fr_in_itv', 15
    });

n_tr = size(ev, 1);
n_fr = size(ev, 2);
n_itv = ceil(n_fr / S.n_fr_in_itv);
ev_itv = nan(n_tr, n_itv);

for i_itv = 1:n_itv
    fr_st = (i_itv - 1) * S.n_fr_in_itv + 1;
    fr_en = min(i_itv * S.n_fr_in_itv, n_fr);
    fr_incl = fr_st:fr_en;
    
    ev_itv(:, i_itv) = nanmean(ev(:, fr_incl), 2);
end
end