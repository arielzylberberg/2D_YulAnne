function [sTr, S, Pred] = pred_en_simple(varargin)
% Can be run on its own; just change 'policy'

S = varargin2S(varargin, {
    'n_tr_per_cond', 1000
    'seed', 0
    'subj', ''
    'policy', 'TargNorm' % 'Serial'
    'conds0', {{-0.001, 0, 0.001}, {-0.001, 0, 0.001}}
    });

%%
rng(S.seed);

n_dim = 2;
n_ch = 2;

conds0 = S.conds0;
conds = cell2mat(factorize(conds0));
n_conds_all = size(conds, 1);

ens0 = cell(n_conds_all, n_dim);
conds0 = cell(n_conds_all, n_dim);

en00 = cell(1, n_dim);

for dim = 1:n_dim
    for i_cond = 1:n_conds_all
        cond1 = conds(i_cond, dim);

        Sim = SimEn('n_tr', S.n_tr_per_cond);
        en1 = Sim.main;
        en1 = (en1(:,:,dim) + cond1) * 100;
        ens0{i_cond, dim} = en1;
        conds0{i_cond, dim} = zeros(size(en1, 1), 1) + cond1;
    end
    en00{1, dim} = cat(1, ens0{:,dim});
end
cond = cell2mat(conds0);
en = cat(3, en00{:});

n_tr = size(en, 1);
n_fr = size(en, 2);
bound = ones(n_tr, n_fr, n_dim) ...
    .* reshape2vec([1.4, 0.7], 3);
gain = ones(n_tr, n_fr, n_dim);

if isempty(S.subj)
    subj = ['Sim', S.policy];
    S.subj = subj;
else
    subj = S.subj;
end

%%
switch S.policy
    case 'Serial'
        Pred = PredEnSerial;
    case 'Parallel'
        Pred = PredEnParallel;
    case 'ParAcqSerResp'
        Pred = PredEnParAcqSerResp;
    case 'SerMultSwitch'
        Pred = PredEnSerMultSwitch;
    case 'TargNorm'
        Pred = PredEnTargNorm;
    otherwise
        error('Unknown policy: %s\n', S.policy);
end
W = Pred;

%%
[ch, rt, res, ens0] = Pred.pred(en, bound);

sTr = table;
z = zeros(n_tr, 1);
sTr.subj = repmat({subj}, [n_tr, 1]);
sTr.n_dim_task = z + 2;

sTr.en = cat(2, ens0{:});
sTr.ch = ch;
sTr.cond = cond;
sTr.RT = rt;

sTr.dim_rel = [z, z] + 1;
for dim_rel = 1:n_dim
    dim_irr = n_dim + 1 - dim_rel;
    [~,~,sTr.dif_rel(:, dim_rel)] = unique(abs(cond(:, dim_rel)));
    [~,~,sTr.dif_irr(:, dim_rel)] = unique(abs(cond(:, dim_irr)));
    [~, ~, sTr.adcond(:,dim)] = unique(abs(sTr.cond(:, dim)));
    [~, ~, sTr.dcond(:,dim)] = unique(sTr.cond(:, dim));
end

end