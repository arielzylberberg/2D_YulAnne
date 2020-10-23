% main_merge_RT_sh_VD_bi
% merge uni/bimanual into RT_sh_VD

clear

L1 = load('../../Data_2D/sTr/combined_2D_RT_sh_VD.mat');
L2 = load('../../Data_2D/orig_Anne_uni_bimanual_RT/energy_Annes_mono_bimanual.mat');

%%
% L1 = 
%   struct with fields:
%             RT: [86496�1 double]
%          accuC: [86496�1 double]
%          accuM: [86496�1 double]
%             ch: [86496�2 double]
%           cond: [86496�2 double]
%        dim_rel: [86496�2 logical]
%             en: {86496�2 cell}
%      i_all_Run: [86496�1 double]
%       id_parad: [86496�1 double]
%        id_subj: [86496�1 double]
%     n_dim_task: [86496�1 double]
%          parad: []
%         parads: {10�1 cell}
%           subj: []
%          subjs: {6�1 cell}
%      t_RDK_dur: [86496�1 double]
%           task: [86496�1 char]
%        to_excl: [86496�1 logical]
%           xyct: {86496�1 cell}

% L2 =
%   struct with fields:
%                         E: [37237�395 double]
%                  RT_color: [37237�1 double]
%                 RT_motion: [37237�1 double]
%                       RTs: [37237�2 double]
%                  bimanual: [37237�1 logical]
%              choice_color: [37237�1 logical]
%             choice_motion: [37237�1 logical]
%      color_fraction_frame: [37237�375 double]
%     color_responded_first: [37237�1 double]
%                       dat: []
%                     group: [37237�1 double]
%     motion_fraction_frame: [37237�375 double]
%          signed_color_coh: [37237�1 double]
%         signed_motion_coh: [37237�1 double]
%                      time: [1�375 double]
%                   time_me: [1�395 double]

%%
L12 = L1;
L12.parad = L12.parads(L12.id_parad);
L12.subj = L12.subjs(L12.id_subj);

%%
n_tr0 = size(L1.RT, 1);
n_tr = size(L2.RTs, 1);

L22 = struct;

L22.RT = max(L2.RTs, [], 2); % max RT
L22.RTs = [L2.RT_motion, L2.RT_color];
L12.RTs = nan(n_tr0, 2);


L22.ch = [L2.choice_motion + 1, L2.choice_color + 1];
L22.cond = [L2.signed_motion_coh, L2.signed_color_coh];
L22.dim_rel = true(n_tr, 2);

L12.bimanual = false(n_tr0, 1);
L22.bimanual = L2.bimanual;

L22.en = cat(3, L2.E(:,1:375), L2.color_fraction_frame);

L22.i_all_Run = ones(n_tr, 1); % not correct - a placeholder

L22.parad = repmat({'unibimanual'}, [n_tr, 1]);
L22.subj = arrayfun(@(v) sprintf('ID%d', v), L2.group, ...
    'UniformOutput', false);

L22.t_RDK_dur = L22.RT;
L22.task = repmat('A', [n_tr, 1]);
L22.to_excl = false(n_tr, 1);
L22.n_dim_task = 2 + zeros(n_tr, 1);

% L3.time = L2.time;
% L3.time_me = L2.time

ens = cell(1, 2);
for dim = 1:2
    ens{dim} = cell2mat2(L1.en(:,dim), 'max_len', 375);
end
L12.en = cat(3, ens{:});

%%
L4 = struct;
for field = {
        'RT', 'ch', 'cond', 'dim_rel', 'en', ...
        'i_all_Run', 'parad', 'subj', 'n_dim_task', ...
        'RTs', 'bimanual', ...
        't_RDK_dur', 'task', 'to_excl'
        }
    fprintf('Copying %s\n', field{1});
    L4.(field{1}) = [L12.(field{1}); L22.(field{1})];
end

%%
[L4.subjs, ~, L4.id_subj] = unique(L4.subj);
L4 = rmfield(L4, 'subj');

[L4.parads, ~, L4.id_parad] = unique(L4.parad);
L4 = rmfield(L4, 'parad');

%%
file_out = '../../Data_2D/sTr/combined_2D_RT_sh_VD_unibimanual.mat';
fprintf('Saving to %s\n', file_out);
save(file_out, '-v7', '-struct', 'L4');
fprintf('Done.\n');
