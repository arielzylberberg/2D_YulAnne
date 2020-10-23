function main_plot_lev_t(C0, tbl)

% n_bin_to_pool = 6;
% dif_rels = {1:2};
% dif_irrs = {1, 2}; % {1:2, 3:5};
% aligns = [-1, 1];
% en_fields = {'nnME', 'mCE'};
% % en_fields = {'mME', 'mCE'};
% 
% n_subj = 3;
% n_dim = 2;
% 
% n_dif_irrs = numel(dif_irrs);
% n_align = numel(aligns);

n_subj = 3;
n_dim = 2;

S0 = varargin2S(C0, {
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

% n_bin_to_pool = 12;
% dif_rels = {1:3};
% dif_irrs = {1:3, 4:5};
% aligns = [-1, 1];
% n_subj = 3;
% n_dim = 2;
% 
% n_dif_irrs = numel(dif_irrs);
% n_align = numel(aligns);
% 
% C0 = varargin2C({
%     'subj', num2cell(1:n_subj)
%     'n_dim_task', num2cell(1:n_dim)
%     'dim_rel', num2cell(1:n_dim)
%     'dif_rel', dif_rels
%     'dif_irr', dif_irrs
%     'align', num2cell(aligns)
%     'n_bin_to_pool', n_bin_to_pool
%     'en_fds', {{'nnME', 'mCE'}} 
%     % 'en_fds', {{'mME', 'mCE'}}
%     });

%%
[file, S0] = get_lev_file(C0);

% file = '../Data_2D/a03_En/RevCorSimple/subj={S1,S2,S3}+n_dim_task={1,2}+dim_rel={1,2}+dif_rel={[1,2]}+dif_irr={[1,2],[3,4,5]}+align={-1,1}+n_bin_to_pool=6+en_fds={{nnME,mCE}}+lev={mean}+cum={sum}'
if nargin < 2 || isempty(tbl)
    L1 = load([file, '.mat']);
    fprintf('Loaded %s.mat\n', file);
    tbl = L1.tbl;
end

%% Plotting
Plot = MainPlotRevCorSimple;
for dif_incl1 = S0.dif_rel
    dif_incl = dif_incl1{1};

    for subj = S0.subj(:)' % cell2mat(S0.subj)
        fig_tag(subj);
%         fig_tag(sprintf('S%d', subj));
        clf;
        
        S1 = varargin2S({
            'dif_rel', {dif_incl}
            'subj', {subj}
            }, S0);
        tbl1 = Plot.plot_by_dim_align(tbl, S1);

        pth = '../Data_2D/a03_En/RevCorSimple';
        nam = bml.str.Serializer.convert(varargin2S({
            'plt', 'main_revcor_simple'
            'sbj', subj
            'y', tbl1.lev
            'difch', dif_incl
            'difcurve', S0.dif_irr
            'pool', S0.n_bin_to_pool
            'en_fds', S0.en_fds
            }));
        file = fullfile(pth, nam);
        savefigs(file, 'size', [800, 400]);
    end
end

%%
for ii = 1:4
    subplotRC(2,4,2,ii);
    ylim([0, 0.05]);
end