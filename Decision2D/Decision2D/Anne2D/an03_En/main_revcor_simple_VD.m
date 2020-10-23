% main_revcor_simple_VD
init_path;

%% Load data
file_in = '../Data_2D/sTr/AnneVD.mat';
L = load(file_in);
fprintf('Loaded %s\n', file_in);

%% Settings
subjs = {'ID2', 'ID3'};

n_bin_to_pool = 1;
dif_rels = {1};
dif_irrs = {1, 2, 3};
aligns = [-1, 1];
en_fields = {'en', 'en'};
% en_fields = {'nnME', 'mCE'};
% en_fields = {'mME', 'mCE'};

n_subj = 2;
n_dim = 2;

n_dif_irrs = numel(dif_irrs);
n_align = numel(aligns);

%% Simplify variables
sTr = struct2table(L);
[~,sTr.i_subj] = ismember(sTr.subj, subjs);
sTr.dim_rel(:,1) = ismember(sTr.task, {'A', 'H'});
sTr.dim_rel(:,2) = ismember(sTr.task, {'A', 'V'});
sTr.n_dim_task = strcmp(sTr.task, 'A') + 1;
[~,sTr.i_parad] = ismember(sTr.parad, {'RT', 'sh', 'VD'});

% sTr.en(:,1) = sTr.(en_fields{1});
% sTr.en(:,2) = sTr.(en_fields{2});
sTr.cond = [sTr.condM, sTr.condC];
% sTr.nnME = [];
% sTr.mME = [];
% sTr.mCE = [];
sTr.ch = [sTr.subjM, sTr.subjC] + 1;

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
pth = '../Data_2D/an03_En/RevCorSimple';
Main = MainRevCorSimple;
C0 = varargin2C({
    'subj', subjs % num2cell(1:n_subj)
    'n_dim_task', num2cell(1:n_dim)
    'dim_rel', num2cell(1:n_dim)
    'dif_rel', dif_rels
    'dif_irr', dif_irrs
    'align', num2cell(aligns)
    'n_bin_to_pool', n_bin_to_pool
    'en_fds', {en_fields}
    't_RDK_dur', {0.6:0.12:1.2}
    });
[Ss, S0, file] = Main.get_Ss(C0, 'pth', pth);
mkdir2(pth);
[~, nam] = fileparts(file);
file = fullfile(pth, nam);

%% Compute results and save
tbl = Main.main(sTr, Ss);

%%
save(file, 'S0', 'tbl');
fprintf('Saved to %s.mat\n', file);
