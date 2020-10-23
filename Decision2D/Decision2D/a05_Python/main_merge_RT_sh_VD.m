% main_merge_RT_sh_VD

%%
L1 = load('../../Data_2D/sTr/combined_2D_RT_sh.mat');
L2 = load('../../Data_2D/sTr/AnneVD.mat');

% dataset2struct(L1.dat(1,:))
% 
% ans = 
% 
%   struct with fields:
% 
%                  parad: 'I_RTec6'
%                  repID: 2
%                 condID: 63
%                  succT: 1
%                aborted: 0
%              cancelled: 0
%              attempted: 1
%                   task: 'V'
%                  condC: 1
%                  condM: -0.1280
%                   freq: 1
%                    gap: 0
%            match_1D_2D: 1
%                  seedC: 3.4963e+09
%                  seedM: 2.4131e+09
%               seed_num: 1
%              t_RDK_dur: Inf
%        unique_seed_num: 1
%                   i_Tr: 3
%               i_all_Tr: 3
%                  i_Run: 2
%              i_all_Run: 2
%              condID_Tr: 63
%             condID_Run: 3
%                  corrC: 2
%                  corrM: 1
%                  subjC: 2
%                  subjM: NaN
%                  score: 1
%                     RT: 0.9404
%                    tSt: 1.1638e+04
%                    tEn: 1.1641e+04
%              trialFile: '/Users/yul/Dropbox/CodeNData_2D/Data_2D/Shadlen/+MDD/+expr/MDD.expr.run_exp/I_RTec6/LT/orig/MDD.expr.run_exp_I_RTec6__LT_20140127T131350.mat'
%                runFile: '/Users/yul/Dropbox/CodeNData_2D/Data_2D/Shadlen/+MDD/+expr/MDD.expr.run_exp/I_RTec6/LT/orig/MDD.expr.run_exp_I_RTec6__LT_20140127T131240_debugged run_exp at new_Tr.mat'
%     tLastBeep2Brighten: NaN
%         t_RDK_dur_plan: Inf
%          t_RDK_dur_obs: 0.9611
%         flag_fb_timing: 'good_timing'
%             fr_dropped: 0
%              timestamp: 7.3563e+05
%           bias_irr_b_M: [NaN 0 0]
%          bias_irr_se_M: [NaN 0 0]
%          slope_irr_b_M: [NaN 0 0]
%         slope_irr_se_M: [NaN 0 0]
%           n_back_irr_M: 200
%           bias_irr_b_C: [0 0 0]
%          bias_irr_se_C: [0 0 0]
%          slope_irr_b_C: [0 0 0]
%         slope_irr_se_C: [0 0 0]
%           n_back_irr_C: 0
%                  corrS: 0
%                  subjS: 0
%                 rM_mom: [1�375 double]
%                 rM_cum: [1�375 double]
%                 rM_sum: 0
%                 rC_mom: [1�375 double]
%                 rC_cum: [1�375 double]
%                 rC_sum: 0
%                 rS_mom: [1�0 double]
%                 rS_cum: [1�0 double]
%                 rS_sum: 0
%                    cCE: [1�72 double]
%                    cME: [1�87 double]
%                   crCE: [1�375 double]
%                   crME: [1�375 double]
%                fail_En: 0
%                 has_En: 1
%                has_rEn: 1
%                    mCE: [1�72 double]
%                    mME: []
%                   mrCE: [1�375 double]
%                   mrME: [1�375 double]
%                    nCE: 72
%                    nME: 87
%                   nrCE: 72
%                   nrME: 72
%                    tCE: 3.6457e+03
%                    tME: -1.1759e+04
%                   trCE: 51.7186
%                   trME: -35
%                 meanCE: 50.6345
%                 meanME: -135.1581
%                   xyct: [288�4 double]
%                     En: []
%           ME_oct_st500: -1.0462e+04
%                   nnME: [1�95 double]
%                 i_subj: 1
%            parad_short: 'RT'
%              i_all_sTr: 1
%            is_valid_tr: 0
%               trialID1: 0
%               trialID2: 0
%               trialID3: 0
%               trialID4: 0
%                 seedM1: 0
%                 seedM2: 0
%                 seedC1: 0
%                 seedC2: 0
%                dot_dir: 0
%                  ans_M: 0
%                  coh_M: 0
%                maj_col: 0
%                  ans_C: 0
%                  coh_C: 0
%                  rdkOn: 0
%                 rdkDur: 0
%               rdkOn2Go: 0
%                   goRT: 0
%              ch_mean_M: 0
%               d_cond_M: 0
%              ch_mean_C: 0
%               d_cond_C: 0
%              RT_mean_M: 0
%              RT_mean_C: 0
%               t_RDK_on: 0
%                  task0: 0
%                  accuM: 0
%                  accuC: 0
%                  condS: 0
%                    rCE: []
%                    rME: []
%                acondCE: 0
%                acondME: 0
%               ascondCE: 0
%               ascondME: 0
%                 condCE: 0
%                 condME: 0
%               dacondCE: 0
%               dacondME: 0
%              dascondCE: 0
%              dascondME: 0
%                dcondCE: 0
%                dcondME: 0
%               dscondCE: 0
%               dscondME: 0
%                scondCE: 0
%                scondME: 0
%                   stCE: 0
%                   stME: 0
 
% L2 = 
% 
%   struct with fields:
% 
%                RT: [32254�1 double]
%             accuC: [32254�1 double]
%             accuM: [32254�1 double]
%                ch: [32254�2 logical]
%              cond: [32254�2 double]
%             condC: [32254�1 double]
%             condM: [32254�1 double]
%           dim_rel: [32254�2 logical]
%      en_color_LLR: {32254�1 cell}
%     en_color_mean: {32254�1 cell}
%         i_all_Run: [32254�1 double]
%        n_dim_task: [32254�1 double]
%             parad: {32254�1 cell}
%            raw_en: {32254�2 cell}
%              subj: {32254�1 cell}
%             subjC: [32254�1 logical]
%             subjM: [32254�1 logical]
%         t_RDK_dur: [32254�1 double]
%              task: {32254�1 cell}
%           to_excl: [32254�1 logical]
%              xyct: {32254�1 cell}
%                en: {32254�2 cell}

%%
ds1 = L1.dat;
ds2 = struct2dataset(L2);

%%
S1 = dataset2struct(ds1(1,:));
S2 = dataset2struct(ds2(1,:));
bml.struct.compare_structs(S1, S2)

%%
n1 = size(ds1, 1);
n2 = size(ds2, 1);
ix_new = n2 + (1:n1);
% ix_new = n1 + (1:n2);

ds1.ch = [ds1.subjM, ds1.subjC];
ds1.cond = [ds1.condM, ds1.condC];
ds1.dim_rel = [ds1.task ~= 'V', ds1.task ~= 'H'];
ds1.n_dim_task = sum(ds1.dim_rel, 2);
ds1.en = [ds1.nnME, ds1.mCE];
ds1.raw_en = [ds1.rME, ds1.rCE];
ds1.subj = arrayfun(@(v) sprintf('S%d', v), ds1.i_subj, ...
    'UniformOutput', false);
ds1.subj(strcmp(ds1.subj, 'S4')) = {'FR'};
ds1.to_excl = ~ds1.succT;

for fields_to_remove = {
        'subjM', 'subjC', 'condM', 'condC', ...
        'en_color_LLR', 'en_color_mean', ...
        'raw_en'
        }
    try
        ds2.(fields_to_remove{1}) = [];
    catch err
        warning(err_msg(err));
    end
end

ds2.task = char(ds2.task);
ds2.ch = double(ds2.ch);

% ds1 = ds_set(ds1, ix_new, ds2);
for f = setdiff(ds2.Properties.VarNames(:)', ...
        {'en_color_LLR', 'en_color_mean'})
    fprintf('Copying %s \n', f{1});    
    ds2.(f{1})(ix_new,:) = ds1.(f{1});
%     ds1.(f{1})(ix_new,:) = ds2.(f{1});
end

%%
ds1.parad(cellfun(@isempty, ds1.parad)) = {''};
ds2.parad(cellfun(@isempty, ds2.parad)) = {''};
ds1.subj(cellfun(@isempty, ds1.subj)) = {''};
ds2.subj(cellfun(@isempty, ds2.subj)) = {''};

dat = dataset2struct(ds2, 'AsScalar', true);

%%
[parads, ~, id_parad] = unique(dat.parad);
[subjs, ~, id_subj] = unique(dat.subj);

dat.parads = parads;
dat.id_parad = id_parad;
dat.parad = [];

dat.subjs = subjs;
dat.id_subj = id_subj;
dat.subj = [];

%%
save('../../Data_2D/sTr/combined_2D_RT_sh_VD.mat', '-struct', 'dat', '-v7');