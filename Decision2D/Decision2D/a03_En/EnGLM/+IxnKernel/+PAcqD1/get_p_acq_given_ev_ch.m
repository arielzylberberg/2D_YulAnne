function p_acq = get_p_acq_given_ev_ch(ev_itv1, ch_data, params_ch_a, d_cond)
% p_acq = get_p_acq_given_ev_ch(ev_itv1, ch_data, params_ch_a, d_cond)
%
% INPUT:
% ev_itv1(tr, 1)
% : momentary evidence on s-th interval
%
% ch_data(tr, [ch0, ch1]): 0 or 1
%
% params_ch_given_a
% = params_ch{a}
% = ([beta0, beta1])
% : a: number of intervals acquired before s
%
% OUTPUT:
% p_acq(1, itv)
%
% See Yul Kang 2018, PhD thesis, Figure 6.2

f_p_ch_given_p_acq = @(p_acq) ...
    -IxnKernel.PAcqD1.get_log_p_ch_given_p_acq( ...
        p_acq, ev_itv1, ch_data, params_ch_a, d_cond);

p_acq = fmincon( ...
f_p_ch_given_p_acq, 0.5, [], [], [], [], ...
0, 1);

%%
fprintf('in get_p_acq_given_ev_ch:');
disp(f_p_ch_given_p_acq(0.5)); % DEBUG

%%
p_acqs = 0:0.1:1;
p_ch = arrayfun(f_p_ch_given_p_acq, p_acqs);
plot(p_acqs, p_ch, 'o');
disp(p_acq);
end
