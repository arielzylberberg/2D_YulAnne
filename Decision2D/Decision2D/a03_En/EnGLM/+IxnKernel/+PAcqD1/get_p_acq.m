function [p_acq_given_s, info] = get_p_acq(ev, ch_data, ch_pred, cond, varargin)
% [p_acq_given_s, info] = get_p_acq(ev, ch_data, ch_pred, cond, ...)
%
% INPUT:
% ev(tr, fr)
% ch_data(tr, [ch0, ch1]): each row sums to 1
% : logical matrix indicating the observed choice.
% ch_pred(tr, [ch0, ch1]): each row sums to 1
% : predicted probability of each choice on each trial,
%   if all evidence is always acquired.
% cond(tr, 1)
%
% OPTION:
% 'n_itv', 5
% 'dif_incl', 1:2 % 1 is hardest (smallest absolute cond)
% 'truncate_st_fr', 15 % # frames to ignore at the beginning
% 'n_dt_in_itv', 10 % # frames go into one "interval" or "chunk"
%
% OUTPUT:
% p_acq_given_s(s)
% info: struct
% .params_ch{itv, d_cond}([beta0, beta1])
% .p_acq_given_a_s(a, s)
% .p_a_given_s(a, s)
% .p_acq_a_given_s(a, s)
% .p_acq_given_s(1, s)
% .s: 1:n_itv
% .a: 0:(n_itv - 1)
% .S : struct of options
%
% See Yul Kang 2018, PhD Thesis, Chapter 6.
% Augmented by pooling across conditions.

% 2017 (c) Yul Kang, yul dot kang dot on at gmail dot com.

S = varargin2S(varargin, {
    'n_itv', 5
    'dif_incl', 1:2 % 1 is hardest (smallest absolute cond)
    'truncate_st_fr', 15 % # frames to ignore at the beginning
    'n_fr_in_itv', 10 % # frames go into one "interval" or "chunk"
    });

%% Preprocess
truncate_st_fr = S.truncate_st_fr;
ev = ev(:, truncate_st_fr:end);

%% Filter
dif_incl = S.dif_incl;

[~, ~, ad_cond] = unique(abs(cond));
tr_incl_dif = ismember(ad_cond, dif_incl);

ch_data = ch_data(tr_incl_dif, :);
ch_pred = ch_pred(tr_incl_dif, :);
cond = cond(tr_incl_dif, :);
ev = ev(tr_incl_dif, :);

assert(iscolumn(cond), 'cond must be a column vector!');
[~, ~, d_cond] = unique(cond);
n_cond = max(d_cond);

%% For each d_cond...
%% Get ev_itv
% ev_itv: previously ev_summary_interval
ev_itv = IxnKernel.PAcqD1.summarize_ev(ev, ...
    'n_fr_in_itv', S.n_fr_in_itv);

%% Get params_ch
n_itv = S.n_itv;
params_ch = cell(n_itv, n_cond); % {a+1, d_cond}([beta0, beta1])

for d_cond1 = 1:n_cond
    tr_incl = d_cond == d_cond1;
    
    for a = 0:(n_itv - 1)
        params_ch{a+1, d_cond1} = IxnKernel.PAcqD1.get_params_ch( ...
            ev_itv(tr_incl, a+1), ch_pred(tr_incl, :));
    end
end

%% DEBUG: display params_ch
for a = 0:(n_itv - 1)
    for d_cond1 = 1:n_cond
        fprintf(' %+1.1f', params_ch{a+1, d_cond1});
        fprintf('    ');
    end
    fprintf('\n');
end

%% Get p_acq_a_s
p_acq_given_a_s = zeros(n_itv, n_itv); % (a+1, s)
for s = 1:n_itv
    for a = 0:(s - 1)
        p_acq_given_a_s(a+1, s) = ...
            IxnKernel.PAcqD1.get_p_acq_given_ev_ch( ...
                ev_itv(:, s), ch_data, params_ch(a+1, :), d_cond);
    end
end

%% Get p_a_given_s
p_a_given_s = IxnKernel.PAcqD1.get_p_a_given_s(p_acq_given_a_s);

%% Get p_acq_given_s
p_acq_a_given_s = p_acq_given_a_s .* p_a_given_s;
p_acq_given_s = sum(p_acq_a_given_s);

%% Wrap output
s = 1:n_itv;
a = s - 1;

info = packStruct( ...
    params_ch, ...
    p_acq_given_a_s, p_a_given_s, p_acq_a_given_s, p_acq_given_s, ...
    s, a, S);
end