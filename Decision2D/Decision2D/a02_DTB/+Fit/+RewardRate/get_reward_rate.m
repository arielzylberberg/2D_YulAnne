function [reward_rate, mean_RT, mean_accu, info] = ...
    get_reward_rate(data, pred, t, mean_iti)
% data(t, cond1, cond2, ch1, ch2)
% pred(t, cond1, cond2, ch1, ch2)
% t: vector of time in seconds
%
% data: in the unit of number of trials
% pred: conditional probability within each (cond1, cond2) pair
%       i.e., all(sums(pred, [1, 4, 5])) = 1

n_dim = 2;
n_cond = sizes(pred, [2, 3]);
ch_accu_dim = cell(1, n_dim);

for dim = 1:2
    n_cond1 = n_cond(dim);
    mid_cond = round((n_cond1 + 1) / 2);
    conds = 1:n_cond1;
    ch_accu_dim{dim} = [
        (conds < mid_cond) + (conds == mid_cond) * 0.5
        (conds > mid_cond) + (conds == mid_cond) * 0.5
        ]';
end
ch_accu = bsxfun(@times, ...
    permute(ch_accu_dim{1}, [3, 1, 4, 2]), ...
    permute(ch_accu_dim{1}, [3, 4, 1, 5, 2]));

n_tr_cond_data = sums(data, [1, 4, 5]);
n_tr_pred = bsxfun(@times, n_tr_cond_data, pred);

total_reward = sums( ...
    bsxfun(@times, ch_accu, n_tr_pred), ...
    [1, 4, 5], true);
total_RT = sums( ...
    bsxfun(@times, t, n_tr_pred), ...
    [1, 4, 5], true);
total_iti = squeeze(mean_iti * n_tr_cond_data);

reward_rate_cond = total_reward ./ (total_RT + total_iti);
reward_rate = sum(total_reward(:)) ...
    ./ (sum(total_RT(:)) + sum(total_iti(:)));

mean_RT = sum(total_RT(:)) / sum(n_tr_cond_data(:));
mean_accu = sum(total_reward(:)) / sum(n_tr_cond_data(:));

info = packStruct( ...
        reward_rate_cond, ...
        total_reward, ...
        total_RT, ...
        total_iti, ...
        n_tr_cond_data, ...
        n_tr_pred, ...
        ch_accu);
end
