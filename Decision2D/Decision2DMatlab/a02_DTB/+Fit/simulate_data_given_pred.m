function [data, ch_new, rt_vec_new] = ...
    simulate_data_given_pred(pred, dCond, t, varargin)
% data = simulate_data_given_pred(pred, varargin)
%
% INPUT:
% pred(t, cond1, cond2, ch1, ch2)
% dCond(tr, dim) = dCond % 1 to nCond(dim)
%
% OUTPUT:
% data(t, cond1, cond2, ch1, ch2)

S = varargin2S(varargin, {
    'seed', 1
    });

%%
rng(S.seed);

n_cond1 = size(pred, 2);
n_cond2 = size(pred, 3);
% n_cond1 = W.Data.nConds(1);
% n_cond2 = W.Data.nConds(2);
n_ch = 2;
n_tr = size(dCond, 1);
% dCond = W.Data.get_dCond;
% t = W.t;
nt = numel(t);
% ch = W.Data.ch;
% rt_vec = W.Data.rt;
t_ch = (1:(nt * n_ch^2))';

data = zeros(size(pred));

ch_new = zeros(n_tr, n_ch);
rt_vec_new = zeros(n_tr, 1);

for dCond1 = 1:n_cond1
    for dCond2 = 1:n_cond2
        % To keep only condition frequencies in the bootstrap
        % (right thing to do)
        tr_incl = (dCond(:,1) == dCond1) ...
                & (dCond(:,2) == dCond2);
        n_tr_incl = nnz(tr_incl);

        if n_tr_incl > 0
            rt_pdf1 = vVec(pred(:, dCond1, dCond2, :, :));
            rt_ch_vec = ...
                randsample(t_ch, n_tr_incl, true, rt_pdf1);

            [rt_ix, ch1, ch2] = ind2sub([nt, n_ch, n_ch], rt_ch_vec);
            rt_vec_new(tr_incl) = t(rt_ix);
            ch_new(tr_incl, :) = [ch1, ch2];

            RT_data_pdf1 = accumarray([rt_ix, ch1, ch2], 1, ...
                [nt, n_ch, n_ch], @sum);

            data(:, dCond1, dCond2, :, :) = ...
                reshape(RT_data_pdf1, [nt, 1, 1, n_ch, n_ch]);
        end

%             % To keep condition & choice frequencies in the bootstrap
%             for ch1 = 1:n_ch
%                 for ch2 = 1:n_ch
%                     tr_incl = (cond(:,1) == cond1) ...
%                             & (cond(:,2) == cond2) ...
%                             & (ch(:,1) == ch1) ...
%                             & (ch(:,2) == ch2);
%                     n_tr_incl = nnz(tr_incl);
% 
%                     if n_tr_incl > 0
%                         rt_pdf1 = pred(:, cond1, cond2, ch1, ch2);
%                         rt_vec(tr_incl) = randsample(t, n_tr_incl, true, rt_pdf1);
%                     end
%                 end
%             end
    end
end