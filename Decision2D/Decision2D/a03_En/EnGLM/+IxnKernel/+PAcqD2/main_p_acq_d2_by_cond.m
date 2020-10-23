clear;
init_path;

%% == Load 2D data
subjs = Data.Consts.subjs_RT(1:3); % (1:3); % (4); % 
n_subj = numel(subjs);
parad = 'RT'; % 'sh'; % 

dats0 = cell(n_subj, 1);
for i_subj = 1:n_subj
    subj = subjs{i_subj};
    file_dat = fullfile('../Data_2D/sTr', sprintf('%s_%s.mat', ...
        parad, subj));
    L0 = load(file_dat);
    dats0{i_subj} = L0.dat;
    fprintf('Loaded %s\n', file_dat);
end

%% == Options to filter data
dif_irr_easy = [-1, 0];
dif_irr_hard = 1:2; % :2;

dif_rel_incl = 1:2; % :2; % For both dims
% n_conds = numel(unique([(dif_rel_incl(:) - 1); -(dif_rel_incl(:) - 1)]));

n_dim_task = 2; % 2

ME_kind = 'nonneg'; % 'octant'|'filterbank'|'nonneg'
pacq_kind = 'slope'; % 'lapse'|'slope'
align_fr = 'en'; % 'mid';
dur_itv_sec = 0.48; % 0.48; % 5; % 0.5;

perm_method = 'rperm'; % 'rperm'|'set0'

truncate_st_fr = 12; % 12; % 0; % 
sum_p_perm = 1; % 0.8;

n_boot = 1; % 100; % 200; % 100;
p_acq_sim1 = (min(sum_p_perm, 1):-0.1:max(sum_p_perm - 1, 0))';
to_randperm = [
    0, 0
%     p_acq_sim1, flip(p_acq_sim1)
    ];
n_randperm = size(to_randperm, 1);

%% File name
pth = '../Data_2D/IxnKernel.PAcqD2.main_p_acq_d2_by_cond';
S = varargin2S({
    'sbj', subjs
    'prd', parad
    't0', align_fr
    'dri', dif_rel_incl
    'ndt', n_dim_task
    'dfie', dif_irr_easy
    'dfih', dif_irr_hard
    'trst', truncate_st_fr
    'titv', dur_itv_sec * 1e3
    'me', ME_kind
    'pacq', pacq_kind
    'pprm', sum_p_perm
    'pmtd', perm_method
    });
name = bml.str.Serializer.convert(S);
file = fullfile(pth, name);

%% == Filter data
n_dim = 2;
% dats = cell(n_subj, n_dim); % {subj, dim_rel, n_dim_task}
ens = cell(n_subj, n_dim, n_dim); % {subj, dim_rel, n_dim_task}(tr, fr)
chs = cell(n_subj, n_dim, n_dim); % {subj, dim_rel, n_dim_task}(tr, dim_ch)
conds = cell(n_subj, n_dim, n_dim); % {subj, dim_rel, n_dim_task}(tr, dim_cond)
ad_conds = cell(n_subj, n_dim, n_dim); % {subj, dim_rel, n_dim_task}(tr, dim_cond)
tasks = Data.Consts.tasks;

for i_subj = 1:n_subj
    dat = dats0{i_subj};

    [~,~,difs{1}] = unique(abs(dat.condM));
    [~,~,difs{2}] = unique(abs(dat.condC));
    switch ME_kind
        case 'octant'
            % TODO: Make it depend on
            % align_fr, dur_itv_sec, and truncate_st_fr.
            
            if truncate_st_fr ~= 0
                warning('truncate_st_fr=%d is ignored for now for octant', ...
                    truncate_st_fr);
            end
            assert(strcmp(align_fr, 'st'));
            assert(dur_itv_sec == 0.5);
            
            ens1{1} = dat.ME_oct_st500;
        case 'filterbank'
            ens1{1} = dat.mME;
            
        case 'nonneg'
            ens1{1} = dat.nnME;
    end
    ens1{2} = dat.mCE;
    conds1 = [dat.condM, dat.condC];
    chs1 = [dat.subjM, dat.subjC];

    %% 1D
    for n_dim_task1 = 1:n_dim
        for dim_rel = 1:n_dim
            if n_dim_task1 == 1
                tr_incl = dat.task == tasks{n_dim_task1, dim_rel};
%                 tr_incl = ismember(difs{dim_rel}, dif_rel_incl) ...
%                         & dat.task == tasks{n_dim_task, dim_rel};
            else
                tr_incl = dat.task == 'A';
%                 tr_incl = ismember(difs{1}, dif_rel_incl) ...
%                         & ismember(difs{2}, dif_rel_incl) ...
%                         & dat.task == 'A';
            end

            en1 = cell2mat2(ens1{dim_rel}(tr_incl, :));
            
            if dim_rel == 1 && strcmp(ME_kind, 'octant')
                ens{i_subj, dim_rel, n_dim_task1} = ...
                    row2cell2(en1);
            else                    
                ens{i_subj, dim_rel, n_dim_task1} = ...
                    row2cell2(en1(:, (truncate_st_fr+1):end));
            end
            conds{i_subj, dim_rel, n_dim_task1} = conds1(tr_incl, :);
            for dim = 1:n_dim
                [~,~,ad_conds{i_subj, dim_rel, n_dim_task1}(:, dim)] = ...
                    unique(abs( ...
                        conds{i_subj, dim_rel, n_dim_task1}(:, dim)));
            end
            chs{i_subj, dim_rel, n_dim_task1} = chs1(tr_incl, dim_rel) == 2;
        end
    end
end

%% == Summarize en
n_fr_per_itv = round(dur_itv_sec * 75);

en_itv = cell(size(ens));
en_itv_all = cell(size(ens));

for i_subj = 1:n_subj
    for dim = 1:n_dim
        for n_dim_task1 = 1:n_dim
            en1 = ens{i_subj,dim,n_dim_task1};
            en_itv1 = cell(size(en1));
            for jj = 1:numel(en1)
                en11 = en1{jj};        
                if dim == 1 && strcmp(ME_kind, 'octant')
                    en_itv11 = en11;
                else
                    n_en11 = length(en11);
                    switch align_fr
                        case 'st'
                            en_itv11 = nanmean(en11(1:min(n_fr_per_itv, n_en11)));
                        case 'en'
                            en_itv11 = nanmean(en11( ...
                                max(n_en11 - n_fr_per_itv + 1, 1):n_en11));
                        case 'mid'
                            en_itv11 = nanmean(en11( ...
                                max(floor((n_en11 - n_fr_per_itv) / 2) + 1, 1) ...
                                :min(floor((n_en11 - n_fr_per_itv) / 2) + n_fr_per_itv, ...
                                     n_en11)));
                        case 'all'
                            en_itv11 = nanmean(en11);
                    end
                end
                en_itv1{jj} = en_itv11;
            end
            en_itv{i_subj,dim,n_dim_task1} = cell2mat2(en_itv1);
        end
    end
end

%%
dim_mot = 1;
if strcmp(ME_kind, 'octant')
    max_ME = 1; % 0.05; % 1.5;
    for i_subj = 1:n_subj
        en_itv{i_subj, dim_mot, n_dim_task} = ...
            max(min(standardize(en_itv{i_subj, dim_mot, n_dim_task}), max_ME), -max_ME);
    end    
elseif strcmp(ME_kind, 'nonneg')
    en_itv{i_subj, dim_mot, n_dim_task} = ...
        standardize(en_itv{i_subj, dim_mot, n_dim_task});
elseif ~isscalar(dif_rel_incl)
    max_ME = 3.5; % 0.05; % 1.5;
    for i_subj = 1:n_subj
        en_itv{i_subj, dim_mot, n_dim_task} = ...
            max(min(en_itv{i_subj, dim_mot, n_dim_task}, max_ME), -max_ME);
    end    
elseif ~isequal(subjs, {'FR'}) && ~strcmp(parad, 'sh') ...
        && isscalar(dif_rel_incl)
    max_ME = 1.5; % 0.05; % 1.5;
    for i_subj = 1:n_subj
        en_itv{i_subj, dim_mot, n_dim_task} = ...
            max(min(en_itv{i_subj, dim_mot, n_dim_task}, max_ME), -max_ME);
    end
end


% % t0 = st
% en_itv = cellfun(@(c) nanmean(c(:, 1:n_fr_per_itv), 2), ...
%     ens, 'UniformOutput', false);

% % t0 = en
% en_itv = cellfun(@(c) nanmean(c(:, end - (1:n_fr_per_itv) + 1), 2), ...
%     ens, 'UniformOutput', false);

% % t0 = mid
% en_itv = cellfun(@(c) nanmean(c(:, ...
%         floor((end - n_fr_per_itv) / 2) + (1:n_fr_per_itv) + 1), 2), ...
%     ens, 'UniformOutput', false);

%% == Identify base (easy) and target (hard) conditions
siz0 = size(ens);
chs_easy = cell(siz0);
chs_hard = cell(siz0);
ens_easy = cell(siz0);
ens_hard = cell(siz0);

% fig_tag(sprintf('%s_%s_%s', ...
%     align_fr, ...
%     sprintf('_%d', dif_irr_easy), ...
%     sprintf('_%d', dif_irr_hard)));
clf;

clear res_hard res_easy p_acq
ax = gobjects(n_subj, n_dim);

p_acq_boot = zeros(n_boot, n_dim, n_subj, n_randperm);
b_easy_boot = zeros(n_boot, n_dim, n_subj, n_randperm);
b_hard_boot = zeros(n_boot, n_dim, n_subj, n_randperm);
se_easy_boot = zeros(n_boot, n_dim, n_subj, n_randperm);
se_hard_boot = zeros(n_boot, n_dim, n_subj, n_randperm);

fig_tag('ch');
for i_subj = n_subj:-1:1
    fprintf('Processing subj %d\n', i_subj);
    for ir = n_randperm:-1:1
        fprintf('Processing randperm %d\n', ir);
        to_randperm1 = to_randperm(ir,:);
        
        for dim_rel = n_dim:-1:1
            dim_irr = n_dim + 1 - dim_rel;

            n_tr = size(chs{i_subj, dim_rel, n_dim_task}, 1);
            [~,~,dcond] = unique( ...
                conds{i_subj, dim_rel, n_dim_task}, 'rows');

            for i_boot = n_boot:-1:1
                if i_boot == 1
                    ix_tr = (1:n_tr)';
                else
                    rng(i_boot);
                    ix_tr = bml.stat.randsample_group(dcond);
                end

                ad_conds_rel1 = ad_conds{i_subj, dim_rel, n_dim_task} ...
                    (ix_tr,dim_rel);
                ad_conds_irr1 = ad_conds{i_subj, dim_rel, n_dim_task} ...
                    (ix_tr,dim_irr);

                ch1 = chs{i_subj, dim_rel, n_dim_task}(ix_tr,:);
                en1 = en_itv{i_subj, dim_rel, n_dim_task}(ix_tr,:);
                cond_rel1 = conds{i_subj, dim_rel, n_dim_task}(ix_tr, dim_rel);
                cond_irr1 = conds{i_subj, dim_rel, n_dim_task}(ix_tr, dim_irr);
                acond_irr1 = abs(cond_irr1);

                n_ad_conds_irr = max(ad_conds_irr1);

                filt_easy = ismember(ad_conds_irr1, dif_irr_easy + n_ad_conds_irr) ...
                          & ismember(ad_conds_rel1, dif_rel_incl);
                filt_hard = ismember(ad_conds_irr1, dif_irr_hard) ...
                          & ismember(ad_conds_rel1, dif_rel_incl);

                ch_easy = ch1(filt_easy);
                ch_hard = ch1(filt_hard);

                en_easy = standardize(en1(filt_easy, :));
                en_hard = standardize(en1(filt_hard, :));

                if ir > 1
                    % Permute easy conditions 
                    % to make surrogate hard condition
                    filt_hard = filt_easy;
                    ch_hard = ch_easy;
                    en_hard = en_easy;
                    
                    cond_rel_perm = cond_rel1(filt_hard);
                    [~,~,dcond_rel_perm] = unique(cond_rel_perm);
                    
                    for icond1 = 1:dcond_rel_perm
                        incl1 = find(dcond_rel_perm == icond1);
                        n_incl1 = numel(incl1);
                        n_perm1 = round(n_incl1 * to_randperm1(dim_rel));
                        ix_perm1 = randperm(n_incl1, n_perm1);
                        
                        switch perm_method
                            case 'rperm'
                                en_hard(incl1(sort(ix_perm1))) = ...
                                    en_hard(incl1(ix_perm1));
                            case 'set0'
                                en_hard(incl1(ix_perm1)) = 0;
                        end
                    end
                    
%                     ix_perm = bml.stat.randperm_group(dcond_rel_hard);
%                     en_hard = en_hard(ix_perm);
                end

                roni_easy = [cond_rel1(filt_easy), ...
                             cond_irr1(filt_easy), ...
                             acond_irr1(filt_easy)];
                roni_hard = [cond_rel1(filt_hard), ...
                             cond_irr1(filt_hard), ...
                             acond_irr1(filt_hard)];

                is_const = @(v) all(bsxfun(@eq, v(1,:), v), 1);
                roni_easy_incl = ~is_const(roni_easy);
                roni_hard_incl = ~is_const(roni_hard);
                roni_easy = roni_easy(:, roni_easy_incl);
                roni_hard = roni_hard(:, roni_hard_incl);

                [p1, ci1, res_hard1, res_easy1] = ...
                        IxnKernel.PAcqD2.get_p_acq_by_cond( ...
                            en_hard, ch_hard, ...
                            en_easy, ch_easy, ...
                            roni_easy, roni_hard, ...
                            'pacq_kind', pacq_kind);
    %                         roni_easy, roni_hard);

                p_acq_boot(i_boot, dim_rel, i_subj, ir) = p1;
                b_easy_boot(i_boot, dim_rel, i_subj, ir) = res_easy1.b(2);
                b_hard_boot(i_boot, dim_rel, i_subj, ir) = res_hard1.b(2);
                se_easy_boot(i_boot, dim_rel, i_subj, ir) = res_easy1.se(2);
                se_hard_boot(i_boot, dim_rel, i_subj, ir) = res_hard1.se(2);

                if i_boot == 1
                    p_acq(i_subj, dim_rel, n_dim_task) = p1;
                    ci_p_acq{i_subj, dim_rel, n_dim_task} = ci1;
                end
            end
            
            if ir > 1
                continue;
            end

            res_hard(i_subj, dim_rel, n_dim_task) = res_hard1;
            res_easy(i_subj, dim_rel, n_dim_task) = res_easy1;

            chs_easy{i_subj, dim_rel, n_dim_task} = ch_easy;
            chs_hard{i_subj, dim_rel, n_dim_task} = ch_hard;

            ens_easy{i_subj, dim_rel, n_dim_task} = en_easy;
            ens_hard{i_subj, dim_rel, n_dim_task} = en_hard;

            %% Plot easy
            ax(i_subj, dim_rel) = subplotRC(n_subj, n_dim, i_subj, dim_rel);

            [h_data, ~, x] = bml.stat.plot_binned_ch(en_easy, ch_easy, ...
                'plot_args', {
                    'LineStyle', 'none'
                    'Marker', 'o'
                    'MarkerFaceColor', 'b'
                    'MarkerEdgeColor', 'w'
                });
            x_pred = linspace(nanmin(x), nanmax(x))';
            switch pacq_kind
                case 'lapse'
                    y_pred = bml.stat.glmval_lapse( ...
                        res_easy1.b([1:2, end]), x_pred);
                case 'slope'
                    y_pred = glmval( ...
                        res_easy1.b(1:2), x_pred, 'logit');
            end
    %         y_pred = glmval(res_easy1.b, x_pred, 'logit');
            hold on;
            h_easy = plot(x_pred, y_pred, 'b-');
            hold on;
            uistack(h_data, 'top');

            %% Plot hard
            [h_data, ~, x] = bml.stat.plot_binned_ch(en_hard, ch_hard, ...
                'plot_args', {
                    'LineStyle', 'none'
                    'Marker', 'o'
                    'MarkerFaceColor', 'r'
                    'MarkerEdgeColor', 'w'
                });

            x_pred = linspace(nanmin(x), nanmax(x))';
            switch pacq_kind
                case 'lapse'
                    y_pred = bml.stat.glmval_lapse( ...
                        res_hard1.b([1:2, end]), x_pred);
                case 'slope'
                    y_pred = glmval(res_hard1.b(1:2), x_pred, 'logit');
            end

            hold on;
            h_hard = plot(x_pred, y_pred, 'r-');
            hold off;
            uistack(h_data, 'top');

            %% Beautify
            ylim([0, 1]);
            bml.plot.beautify;
    %         crossLine('h', 0.5, 'k--');
    %         crossLine('v', 0, 'k--');

            align_fr_label = varargin2S({
                'st',  'initial'
                'mid', 'middle'
                'en',  'final'
                'all', 'all'
                });
            switch align_fr
                case 'all'
                    str_align = '';
                otherwise
                    str_align = sprintf('%s %1.1fs, ', ...
                        align_fr_label.(align_fr), dur_itv_sec);
            end

            if i_subj == 1
                title(sprintf('%s\n ', Data.Consts.dimNames_long{dim_rel}));
            end
            if i_subj == n_subj
                xlabel(sprintf('Average %s\n(%sZ-score)', ...
                    Data.Consts.dimNames_long{dim_rel}, ...
                    str_align));
            end
            str_p_ch = sprintf('P_{ch,%s}', Data.Consts.dimNames{dim_rel});
            if dim_rel == 1
                if i_subj == n_subj
                    ylabel(sprintf('%s\n \n%s', ...
                        subjs{i_subj}, str_p_ch));
                else
                    ylabel(sprintf('%s\n \n ', subjs{i_subj}));
                end
            end
            if dim_rel == 2 && i_subj == n_subj
                ylabel(str_p_ch);
            end
            if i_subj == n_subj
                s_dim = Data.Consts.dimNames{dim_irr};
                legend([h_easy, h_hard], ...
                    {['Easy ', s_dim], ['Hard ', s_dim]}, ...
                    'Location', 'NorthWest');
            else
                set(gca, 'XTickLabel', '');
            end
            if dim_rel == 2
                set(gca, 'YTickLabel', '');
            end
        end
    end
end

%% Combine within subjects
if exist([file '.txt'], 'file')
    delete([file '.txt']);
end
diary([file '.txt']);
 % (boot, dim, subj, perm)
b_hard_boot1 = b_hard_boot ./ b_easy_boot(1,:,:,1);
se_hard_boot1 = se_hard_boot ./ b_easy_boot(1,:,:,1);
w = ones(size(1 ./ se_hard_boot1 .^ 2));
b_hard_mean = squeeze(wmean(b_hard_boot1, w, 2)); % (boot, subj, perm)
% w_mean = squeeze(sum(w, 2));

disp('Slope easy - hard:');
disp(b_easy_boot(1,:,:) - b_hard_boot(1,:,:));

disp('Proportions tried');
disp(to_randperm');

disp('P(perm >= orig) within subject');
disp(squeeze(mean(mean(b_hard_mean(:,:,2:end) >= b_hard_mean(1,:,1), 1), 3)));
disp('P(perm >= orig) within subject (per proportion)');
disp(squeeze(mean(b_hard_mean(:,:,2:end) >= b_hard_mean(1,:,1), 1)));

%% Combine across subjects
siz_comp = [n_boot, n_dim * n_subj, n_randperm];
b_hard_boot_for_pool1 = reshape(b_hard_boot ./ b_easy_boot(1,:,:,1), ...
    siz_comp);
se_hard_boot_for_pool1 = reshape(se_hard_boot ./ b_easy_boot(1,:,:,1), ...
    siz_comp); 
w = ones(size(reshape(1 ./ se_hard_boot_for_pool1 .^ 2, siz_comp)));
b_hard_pool = squeeze(wmean(b_hard_boot_for_pool1, w, 2)); % (boot, perm)

fig_tag('ecdf');
clf;
for i_perm = 2:n_randperm
    ecdf(b_hard_pool(:,i_perm)); hold on;
end
crossLine('v', b_hard_pool(1,1));

disp('P(perm >= orig) across subjects');
disp(mean(b_hard_pool(:,2:end) >= b_hard_pool(1,1), 1));
disp('P(perm >= orig) across subjects (per proportion)');
disp(mean(vVec(b_hard_pool(:,2:end)) >= b_hard_pool(1,1)));

% %% Display results
% p_acq_task = p_acq(:,:,n_dim_task);
% sum_p_acq_task = sum(p_acq_task, 2);
% 
% ci_p_acq_boot = squeeze(prctile(p_acq_boot, [2.5, 97.5], 1));
% 
% sum_p_acq_boot = squeeze(sum(p_acq_boot, 2));
% ci_sum_p_acq_boot = prctile(sum_p_acq_boot, [2.5, 97.5], 1);
% pval1_sum_p_acq = mean(sum_p_acq_boot <= 1, 1);
% pval2_sum_p_acq = mean(sum_p_acq_boot >= 2, 1);

% vprintf(p_acq_task);
% vprintf(ci_p_acq_boot);
% 
% vprintf(sum_p_acq_task);
% vprintf(ci_sum_p_acq_boot);
% vprintf(pval1_sum_p_acq);
% 
% p_combined = invLogit(sum(logit(pval1_sum_p_acq)));
% odds_combined = (1 - p_combined) ./ p_combined;
% vprintf(p_combined);
% vprintf(odds_combined);
% disp(pval2_sum_p_acq);

diary off
fprintf('Saved to %s.txt\n', file);

save_list = bml.file.save_filt(who, {'dats0', 'L0'});
save(file, save_list{:});
fprintf('Saved to %s.mat\n', file);

%% Plot slopes


%% Save figs
fig_tag('ch');
if n_subj == 1
    bml.plot.position_subplots(ax, ...
            'margin_left', 0.17, ...
            'margin_top', 0.06, ...
            'margin_bottom', 0.25, ...
            'btw_col', 0.07);
else
    bml.plot.position_subplots(ax, ...
        'margin_left', 0.17, ...
        'margin_top', 0.06, ...
        'margin_bottom', 0.1, ...
        'btw_col', 0.07);
end
savefigs([file '+plt=ch'], 'size', [400, 50 + n_subj * 150]);

fig_tag('ecdf');
if n_subj == 1
    bml.plot.position_subplots(ax, ...
            'margin_left', 0.17, ...
            'margin_top', 0.06, ...
            'margin_bottom', 0.25, ...
            'btw_col', 0.07);
else
    bml.plot.position_subplots(ax, ...
        'margin_left', 0.17, ...
        'margin_top', 0.06, ...
        'margin_bottom', 0.1, ...
        'btw_col', 0.07);
end
savefigs([file '+plt=ecdf'], 'size', [400, 50 + n_subj * 150]);

%%
% C_ds = [ 
%     {'subj', 'PacqM', 'PacqC', 'sum_PacqM_PacqC', 'pval1'}
%     [subjs(:), ...
%     arrayfun(@(v) sprintf('%1.2f', v), ...
%         [p_acq_task, sum_p_acq_task, pval1_sum_p_acq], ....
%         'UniformOutput', false)]
%     ];
% ds = bml.ds.cell2ds2(C_ds);
% csv_file = [file, '.csv'];
% export(ds, 'File', csv_file, 'Delimiter', ',');

%%
% disp(p_acq_task);
% disp(sum_p_acq_task);
% clear dats0 L0

% txt_file = [file, '.txt'];
% if exist(txt_file, 'file')
%     delete(txt_file);
% end
% 
% diary(txt_file);
% disp(p_acq);
% disp(sum(p_acq, 2));
% diary('off');
% fprintf('Saved to %s\n', txt_file);
















