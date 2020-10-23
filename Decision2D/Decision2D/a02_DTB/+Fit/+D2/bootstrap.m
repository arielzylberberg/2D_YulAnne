function pred = bootstrap(pred0, data, seed)
% bootstrap trials from pred0 according to the trial counts per condition
% in data and using seed if provided
%
% USAGE:
% pred = bootstrap(pred0, data, seed)
%
% INPUT:
% pred0(t, cond1, cond2, ch1, ch2)
% : Probability conditioned on (cond1, cond2)
% data(t, cond1, cond2, ch1, ch2)
% : Number of trials in the bin
% seed: scalar seed number.
%
% OUTPUT:
% pred(t, cond1, cond2, ch1, ch2)
% : Number of simulated trials in the bin, 
%   sampled with replacement using pred0.

if nargin >= 3 && ~isempty(seed)
    rng(seed);
end

%%
nt = size(data, 1);
n_ch = sizes(data, [4, 5]);
n_cond = sizes(data, [2, 3]);
n_cond_all = prod(n_cond);
n_tr_cond_data = sums(data, [1, 4, 5]);
n_tr_cond_vec = hVec(n_tr_cond_data);
pred1 = reshape(permute(pred0, [1, 4, 5, 2, 3]), [], n_cond_all);
n_outcome = size(pred1, 1);
pred = zeros(n_outcome, n_cond_all);

for cond = 1:n_cond_all
    n_tr1 = n_tr_cond_vec(cond);
    w = pred1(:, cond);
    ix = randsample(n_outcome, n_tr1, true, w);
    pred(:,cond) = accumarray(ix, 1, [n_outcome, 1], @sum, 0);
end

pred = permute(reshape(pred, [nt, n_ch, n_cond]), [1, 4, 5, 2, 3]);
end