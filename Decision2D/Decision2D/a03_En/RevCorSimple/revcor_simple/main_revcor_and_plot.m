init_path;

%% Settings
n_bin_to_pool = 6;
dif_rels = {1:2};
dif_irrs = {1:2, 3:5}; % {1:2, 3:5};
aligns = [-1, 1];
en_fields = {'nnME', 'mCE'};
% en_fields = {'mME', 'mCE'};

n_subj = 3;
n_dim = 2;

S0 = varargin2S(varargin, {
    'subj', {'S1', 'S2', 'S3'}  %  num2cell(1:n_subj)
    'n_dim_task', num2cell(1:n_dim)
    'dim_rel', num2cell(1:n_dim)
    'dif_rel', dif_rels
    'dif_irr', dif_irrs
    'align', num2cell(aligns)
    'n_bin_to_pool', n_bin_to_pool
    'en_fds', {en_fields}
    'lev', {'mean'}
    'cum', {'sum'}
    });
C0 = S2C(S0);

%%
[tbl, sTr] = main_revcor_simple(C0{:});

%%
main_plot_lev_t(C0);

%%
pth_out = '../Data_2D/a03_En/main_pred_en_simple';

n_subj = numel(S0.subj);
for i_subj = 1:n_subj
    subj = S0.subj{i_subj};
    y_lims = zeros(2,2,2);
    axs = cell(1, 2);
    for n_dim_task = 1:2
        fig_tag(sprintf('ch_rt_%s_%dD', subj, n_dim_task));
        clf;
        incl = strcmp(sTr.subj, S0.subj{i_subj}) ...
            & sTr.n_dim_task == n_dim_task;
        sTr1 = sTr(incl, :);
        ax = plot_ch_RT(sTr1);
        axs{n_dim_task} = ax;
        
        for dim = 1:2
            y_lims(dim,:,n_dim_task) = ylim(ax(dim,1));
        end        
    end
    for n_dim_task = 1:2
        fig_tag(sprintf('ch_rt_%s_%dD', subj, n_dim_task));
        ax = axs{n_dim_task};
        for dim = 1:2
            ylim(ax(dim,:), [
                min(y_lims(dim,1,:), [], 3), ...
                max(y_lims(dim,2,:), [], 3)
                ]);
        end
        
        file = fullfile(pth_out, bml.str.Serializer.convert(varargin2S({
            'sbj', subj
            'ndim', n_dim_task
            'plt', 'ch_rt'
            })));
        savefigs(file, 'size', [300, 200], 'resolution', 150);
    end
end

%%
n_subj = numel(S0.subj);
for i_subj = 1:n_subj
    subj = S0.subj{i_subj};
    
    y_lims = zeros(2,2,2);
    figs = cell(1, 2);
    for n_dim_task = 1:2
        figs{n_dim_task} = fig_tag(sprintf('ndim%d', n_dim_task));
        figure(figs{n_dim_task});
        clf;
        ax = plot_revcor_simple(tbl, S0, ...
            'subj', subj, ...
            'n_dim_task', n_dim_task);
        gltitle(ax, 'all', sprintf('%s - %dD', subj, n_dim_task), ...
            'shift', [0, 0], ...
            'title_args', {
                'FontSize', 12
                'Position', [0.5, 1.02, 0.5]
                });
            
%         ylim(ax, [-150, 150]);
        xlim(ax(:,1), [0, 2]);
        xlim(ax(:,2), [-2, 0]);

%         C1 = varargin2C({
%             'subj', subj
%             'n_dim_task', n_dim_task
%             }, C0);
%         [~, ~, file] = Main.get_Ss(C1{:});

        for dim = 1:2
            y_lims(dim,:,n_dim_task) = ylim(ax(dim,1));
        end
        axs{n_dim_task} = ax;
    end
    
    for n_dim_task = 1:2
        figure(figs{n_dim_task});
        ax = axs{n_dim_task};
        
        for dim = 1:2
            ylim(ax(dim, :), [min(y_lims(dim, 1, :), [], 3), ...
                              max(y_lims(dim, 2, :), [], 3) * 1.2]);
        end
        
        file = fullfile(pth_out, bml.str.Serializer.convert(varargin2S({
            'sbj', subj
            'ndim', n_dim_task
            'plt', 'lev1D'
            })));
        savefigs(file, 'size', [300, 200], 'resolution', 150);
    end
end
