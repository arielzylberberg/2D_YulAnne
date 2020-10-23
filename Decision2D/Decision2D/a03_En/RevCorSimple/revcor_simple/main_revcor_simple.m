function [tbl, sTr] = main_revcor_simple(varargin)
% init_path;
global L_sTr_eye_all_subj_parad

%% Load data
file_in = '../Data_2D/sTr/sTr_eye_all_subj_parad.mat';
if isempty(L_sTr_eye_all_subj_parad)
    L_sTr_eye_all_subj_parad = load(file_in);
    fprintf('Loaded %s\n', file_in);
end

% n_dif_irrs = numel(dif_irrs);
% n_align = numel(aligns);
% 
%% Simplify variables
n_subj = 3;
n_dim = 2;

S0 = varargin2S(varargin, {
    'subj', {'S1', 'S2', 'S3'}  %  num2cell(1:n_subj)
    'n_dim_task', num2cell(1:n_dim)
    'dim_rel', num2cell(1:n_dim)
    'dif_rel', {1:2}
    'dif_irr', {1:2, 3:5}
    'align', {-1, 1}
    'n_bin_to_pool', 6
    'en_fds', {{'nnME', 'mCE'}}
    'lev', {'mean'}
    'cum', {'sum'}
    });
C0 = S2C(S0);
sTr = struct2table(L_sTr_eye_all_subj_parad);
[~,sTr.i_subj] = ismember(sTr.subj, Data.Consts.subjs_RT);
sTr.dim_rel(:,1) = ismember(sTr.task, {'A', 'H'});
sTr.dim_rel(:,2) = ismember(sTr.task, {'A', 'V'});
sTr.n_dim_task = (sTr.task == 'A') + 1;
[~,sTr.i_parad] = ismember(sTr.parad, {'RT', 'sh'});

sTr.en(:,1) = sTr.(S0.en_fds{1}{1});
sTr.en(:,2) = sTr.(S0.en_fds{1}{2});
sTr.cond = [sTr.condM, sTr.condC];
sTr.nnME = [];
sTr.mME = [];
sTr.mCE = [];
sTr.ch = [sTr.subjM, sTr.subjC];

for i_subj = 1:n_subj
    for dim_rel = 1:n_dim
        tr_incl = sTr.i_subj == i_subj;
        [~,~,dif_rel] = unique(abs(sTr.cond(tr_incl, dim_rel)));
        sTr.dif_rel(tr_incl,dim_rel) = dif_rel;
        
        dim_irr = n_dim + 1 - dim_rel;
        [~,~,dif_irr] = unique(abs(sTr.cond(tr_incl, dim_irr)));
        sTr.dif_irr(tr_incl,dim_rel) = dif_irr;
    end
end

%% Expand settings in a long form
Main = MainRevCorSimple;
[Ss, S0, file] = Main.get_Ss(C0{:});

%% Compute results and save
tbl = Main.main(sTr, Ss);

save(file, 'S0', 'tbl');
fprintf('Saved to %s.mat\n', file);
