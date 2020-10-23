function params_ch = get_params_ch(ev_itv1, ch_pred)
% params_ch = get_params_ch(ev_itv, ch_pred)
%
% INPUT:
% ev_itv1(tr, 1)
% ch_pred(tr, [ch_0, ch_1])
%
% OUTPUT:
% params_ch{a}([beta0, beta1])
% : a: number of intervals acquired before s
%
% See Yul Kang 2018, PhD thesis, Figure 6.2

ch_pred = [ch_pred(:,2), sum(ch_pred, 2)];

params_ch = glmfit(ev_itv1, ch_pred, 'binomial');

plot(ev_itv1, ch_pred(:,1), 'o'); % DEBUG
disp(params_ch);
end