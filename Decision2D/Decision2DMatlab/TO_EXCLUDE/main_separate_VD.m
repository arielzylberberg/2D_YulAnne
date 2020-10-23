% main_separate_VD
%
% Separate 2D VD trials into data_VD.mat

clear;
init_path;

%%
L1 = load('../../2D_YulAnne/data/RT_task/data_RT.mat');

%                             RT: [48209×1 double]
%                            RT1: [48209×1 double]
%     SignedColorStrengthLogodds: [48209×1 double]
%                       bimanual: [48209×1 double]
%                   choice_color: [48209×1 double]
%                  choice_motion: [48209×1 double]
%                      coh_color: [48209×1 double]
%                     coh_motion: [48209×1 double]
%          color_responded_first: [48209×1 double]
%                     corr_color: [48209×1 double]
%                    corr_motion: [48209×1 double]
%                        dataset: [48209×1 double]
%                          group: [48209×1 double]


% % corr_color, corr_motion: whether it was considered correct
% all((L1.corr_motion == (L1.choice_motion == (L1.coh_motion > 0))) | (L1.coh_motion == 0))

%%
L0 = load('../../Data_2D_old/sTr/combined_2D_RT_sh_VD_unibimanual.mat');
%             RT: [123733×1 double]
%             ch: [123733×2 double]
%           cond: [123733×2 double]
%        dim_rel: [123733×2 logical]
%             en: [123733×375×2 double]
%      i_all_Run: [123733×1 double]
%     n_dim_task: [123733×1 double]
%            RTs: [123733×2 double]
%       bimanual: [123733×1 logical]
%      t_RDK_dur: [123733×1 double]
%           task: [123733×1 char]
%        to_excl: [123733×1 logical]
%          subjs: {13×1 cell}
%        id_subj: [123733×1 double]
%         parads: {4×1 cell}
%       id_parad: [123733×1 double]

%%
Lsp = load('../../Data_2D_old/sTr/bimanual_ID4_tdPar_seed1_ef1.mat');
disp(Lsp.dat(1,:));

%% Separate VD from combined - REMOVE
incl = (L0.id_parad == find(strcmp(L0.parads, 'VD'))) ...
    & (L0.task == 'A');
n_incl = sum(incl);

L_VD0 = struct;
for f = setdiff(fieldnames(L0)', {'en', 'subjs', 'parads'})
    L_VD0.(f{1}) = L0.(f{1})(incl, :);
end

L_VD = struct;
L_VD.RT = zeros(n_incl, 1) + nan;
L_VD.RT1 = zeros(n_incl, 1) + nan;
L_VD.SignedColorStrengthLogodds = ...
    logit(L_VD0.cond(:, 2) + 0.5);

L_VD.bimanual = false(n_incl, 1);

L_VD.choice_motion = L_VD0.ch(:, 1);
L_VD.choice_color = L_VD0.ch(:, 2);

L_VD.coh_motion = L_VD0.cond(:, 1);
L_VD.coh_color = L_VD0.cond(:, 2) * 2;
L_VD.coh_color = sign(L_VD.coh_color) .* round(abs(L_VD.coh_color), 4);

L_VD.color_responded_first = false(n_incl, 1);

L_VD.corr_motion = (L_VD0.ch(:, 1) == (L_VD0.cond(:, 1) > 0));
L_VD.corr_color = (L_VD0.ch(:, 2) == (L_VD0.cond(:, 2) > 0));

L_VD.dataset = zeros(n_incl, 1) + 3;
L_VD.group = L_VD0.id_subj;

L_VD.t_RDK_dur = L_VD0.t_RDK_dur;

file_out = '../../data/Var_Dur/data_VD.mat';
if ~exist(fileparts(file_out), 'dir')
    mkdir(fileparts(file_out));
end
save(file_out, '-v7', '-struct', 'L_VD');
fprintf('Saved to %s\n', file_out);

