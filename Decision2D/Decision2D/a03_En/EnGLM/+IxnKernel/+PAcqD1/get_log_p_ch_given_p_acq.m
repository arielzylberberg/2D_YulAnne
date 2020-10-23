function log_p_ch = get_log_p_ch_given_p_acq( ...
    p_acq, ev_itv1, ch_data, params_ch_a, d_cond, varargin)
% log_p_ch = get_log_p_ch_given_p_acq( ...
%     p_acq, ev_itv1, ch_data, params_ch_a)
%
% INPUT:
% p_acq: scalar
% ev_itv1(tr, 1)
% ch_data(tr, [ch0, ch1]): boolean
% params_ch_a([beta0, beta1])
% : When d_cond is given and nonempty, params_ch_a{d_cond}([beta0, beta1])
% d_cond: positive integer index of the condition
%
% OUTPUT:
% log_p_ch: scalar

S = varargin2S(varargin, {
    'ignore_bias_acq', true
    });

if nargin >= 5 && ~isempty(d_cond)
    n_cond = max(d_cond);
    log_p_ch = 0;
    for d_cond1 = 1:n_cond
        tr_incl = d_cond == d_cond1;
        log_p_ch1 = IxnKernel.PAcqD1.get_log_p_ch_given_p_acq( ...
            p_acq, ev_itv1(tr_incl,:), ch_data(tr_incl,:), ...
            params_ch_a{d_cond1});
        log_p_ch = log_p_ch + log_p_ch1;
    end
    return;
end

if S.ignore_bias_acq
    ch_data1 = bsxfun(@rdivide, ch_data, sum(ch_data, 2));
    ch_data1 = [ch_data1(:,2), sum(ch_data1, 2)];
    params_ch_data = glmfit(ev_itv1, ch_data1, 'binomial');
    p_ch2 = glmval([params_ch_data(1); params_ch_a(2)], ev_itv1, 'logit');
else
    p_ch2 = glmval(params_ch_a, ev_itv1, 'logit');
end

p_ch = [1 - p_ch2, p_ch2];

p_ch_unacq = mean(ch_data, 1);
p_ch_given_p_acq = p_ch .* p_acq + p_ch_unacq .* (1 - p_acq);

n_tr = size(ch_data, 1);
p_ch = zeros(n_tr, 1);
p_ch(ch_data(:,2)) = p_ch_given_p_acq(ch_data(:,2), 2);
p_ch(ch_data(:,1)) = p_ch_given_p_acq(ch_data(:,1), 1);

% Ignore missing trials (missing trials are independent of p_acq)
log_p_ch = nansum(log(p_ch)); 