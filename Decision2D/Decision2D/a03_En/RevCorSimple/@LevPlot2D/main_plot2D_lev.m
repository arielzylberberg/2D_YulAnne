% function main_plot2D_lev(Lev)

n_bin_to_pool = 6;
dif_rels = {1:2};
dif_irrs = {1, 2, 3, 4, 5};
aligns = [-1, 1];
n_subj = 3;
n_dim = 2;

%%
C0 = varargin2C({
    'subj', num2cell(1:n_subj)
    'n_dim_task', num2cell(1:n_dim)
    'dim_rel', num2cell(1:n_dim)
    'dif_rel', dif_rels
    'dif_irr', dif_irrs
    'align', num2cell(aligns)
    'n_bin_to_pool', n_bin_to_pool
    });
[file, S0] = get_lev_file(C0{:});
L = load([file, '.mat']);
fprintf('Loaded %s.mat\n', file);
tbl = L.tbl;

%%
n_dim = 2;
n = size(tbl, 1);
dif1 = [tbl.dif_rel, tbl.dif_irr];
tbl.dif = nan(n, n_dim);
tr_incl = tbl.dim_rel == 1;
tbl.dif(tr_incl, :) = dif1(tr_incl, :);
tbl.dif(~tr_incl, :) = flip(dif1(tr_incl, :), 2);

%%
dim_names = {'M', 'C'};
align = -1;
for t = 1:5
    fig_tag(sprintf('t%d', t));
    clf;
    for subj = 1:3
        n_dim_task = 2;

        tr_incl = (tbl.align == align);
        tbl1 = tbl(tr_incl, :);
        ix = [tbl1.subj, tbl1.n_dim_task, tbl1.dim_rel, tbl1.dif];

        levs = cellfun(@(lev1) lev1(t), tbl1.slope, ...
            'ErrorHandler', @(varargin) nan);
        lev = accumarray(ix, levs, [], [], nan);

        errs = cellfun(@(lev1) lev1(t), tbl1.se_slope, ...
            'ErrorHandler', @(varargin) nan);
        err = accumarray(ix, errs, [], [], nan);

%         lev(:,2,:,:,:) = lev(:,2,:,:,:) - lev(:,1,:,:,:);
%         err(:,2,:,:,:) = sqrt(err(:,2,:,:,:).^2 + err(:,1,:,:,:).^2);

        n_dif = 3;
        colors = {cool(n_dif), winter(n_dif)};
        for dim_rel = 1:n_dim

            lev1 = squeeze(lev(subj,n_dim_task,1,:,:));
            lev2 = squeeze(lev(subj,n_dim_task,2,:,:));
            err1 = squeeze(err(subj,n_dim_task,1,:,:));
            err2 = squeeze(err(subj,n_dim_task,2,:,:));
            if dim_rel == 1
                lev1 = lev1';
                lev2 = lev2';
                err1 = err1';
                err2 = err2';
            end
            colors1 = colors{dim_rel};

            ax1 = subplotRC(2,3,dim_rel,subj);
            hs = gobjects(1, n_dif);
            for i_dif = 1:n_dif
                axes(ax1);
        %         hs(i_dif) = ...
        %             plot(lev1(:,i_dif), lev2(:,i_dif), ...
        %                 'o-', 'Color', colors1(i_dif,:));
                hs(i_dif) = bml.plot.errorbar_wo_tick2( ...
                    lev1(:,i_dif), lev2(:,i_dif), ...
                    err1(:,i_dif), [], ...
                    err2(:,i_dif), [], {
                    '-', 'Color', colors1(i_dif,:)
                    });
                hold on;
            end
            legend(hs, csprintf([dim_names{dim_rel}, '%d'], ...
                {1, 2, 3}));
            hold off;

            xlim([-5, 10]);
            ylim([-5, 10]);
        end
    end
%     input(':', 's');
end

% bml.plot.errorbar_wo_tick2

% for subj = cell2mat(S0.subj)
%     for n_dim_task = cell2mat(S0.n_dim_task)
%         subplotRC(n_row, n_col, n_dim_task, subj);
% 
%         for i_dif1 = 1:numel(S0.dif_irr)
%             for i_dif2 = 1:numel(S0.dif_irr)
%                 
%                 incl0 = ...
%                     (L.tbl.subj == subj) ...
%                     & (L.tbl.n_dim_task == n_dim_task);
%                 incl1 = incl0 ...
%                     & (L.tbl.dim_rel == 1) ...
%                     & (L.tbl.dif_irr == S0.dif_irr{i_irr}) ...
%                     & (L.tbl.dif_rel == S0.dif_rel{i_rel});
%                 incl2 = incl0 ...
%                     & (L.tbl.dim_rel == 2) ...
%                     & (L.tbl.dif_irr == S0.dif_irr{i_irr}) ...
%                     & (L.tbl.dif_rel == S0.dif_rel{i_rel});                    
%                     
%         end
%     end
% end