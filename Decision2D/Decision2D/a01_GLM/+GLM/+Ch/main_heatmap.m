%%
init_path;

%%
subjs = Data.Consts.subjs_RT;
n_subj = numel(subjs);

parad = 'RT';
task = 'A';
Ls = cell(1, n_subj);

for i_subj = 1:n_subj
    subj = subjs{i_subj};
    file = fullfile('../Data_2D/sTr', sprintf('%s_%s', parad, subj));
    Ls{i_subj} = load(file);
    fprintf('Loaded %s\n', file);
end

%%
n_dim = 2;
cond = cell(n_subj, 1);
conds = cell(n_subj, n_dim);
dcond = cell(n_subj, 1);
ch = cell(n_subj, 1);
rt = cell(n_subj, 1);
ix_subj = cell(n_subj, 1);

for i_subj = 1:n_subj
    dat = Ls{i_subj}.dat;
    incl = dat.task == task;
    dat = dat(incl, :);
    
    cond1 = [dat.condM, dat.condC];
    dcond1 = zeros(size(cond1));
    for dim = n_dim:-1:1
        [conds{i_subj, dim},~,dcond1(:,dim)] = unique(cond1(:,dim));
    end
    
    ch1 = [dat.subjM, dat.subjC];
    rt1 = dat.RT;
    n_tr1 = size(ch1, 1);
    subj1 = i_subj + zeros(n_tr1, 1);
    
    dcond{i_subj} = dcond1;
    ch{i_subj} = ch1;
    rt{i_subj} = rt1;
    ix_subj{i_subj} = subj1;
end

dcond = cat(1, dcond{:});
ch = cat(1, ch{:});
ix_subj = cat(1, ix_subj{:});
rt = cat(1, rt{:});

cond_ch = accumarray([dcond, ch, ix_subj], 1, [], @sum);
cond_rt = accumarray([dcond, ix_subj], rt, [], @nanmean);

cond_rt_pooled = mean(cond_rt, 3);
    
cond_ch1 = sum(cond_ch, 4);
p_ch{1} = squeeze(cond_ch1(:,:,2,:,:) ./ sum(cond_ch1, 3));

cond_ch1 = sum(cond_ch, 3);
p_ch{2} = squeeze(cond_ch1(:,:,:,2,:) ./ sum(cond_ch1, 4));

p_ch_pooled = cell(1, n_dim);
for dim = 1:n_dim
    p_ch_pooled{dim} = mean(p_ch{dim}, 3);
end

%%
fig_tag('indiv');
ax = subplotRCs(n_dim + 1, n_subj + 1);
for i_subj = 1:(n_subj + 1)
    for dim = 1:(n_dim + 1)
        axes(ax(dim, i_subj));
        if dim == n_dim + 1
            if i_subj == n_subj + 1
                v = cond_rt_pooled';
            else
                v = cond_rt(:,:,i_subj)';
            end
        else
            if i_subj == n_subj + 1
                v = mean(p_ch{dim}, 3)';
            else
                v = p_ch{dim}(:,:,i_subj)';
            end
        end
            
        imagesc(v);
        axis square;
        axis xy;
        set(gca, 'XTick', [], 'YTick', []);
%         ticks_incl = 1:2:9;
%         set(gca, ...
%             'XTick', ticks_incl, ...
%             'YTick', ticks_incl, ...
%             'XTickLabel', conds{i_subj, 1}(ticks_incl) / max(conds{i_subj, 1}), ...
%             'XTickLabelRotation', 45, ...
%             'YTickLabel', conds{i_subj, 2}(ticks_incl) / max(conds{i_subj, 2}));
        bml.plot.beautify;
        
        if dim == 1
            if i_subj == n_subj + 1
                title('Pooled');
            else
                title(sprintf('S%d', i_subj));
            end
        end
        if i_subj == 1
            if dim == 1
                ylabel({'Motion choice', ' ', ' '});
            elseif dim == 2
                ylabel({'Color choice', ' ', ' '});
            else
                ylabel({'RT', ' ', 'Color strength'});
                xlabel('Motion strength');
            end                        
        end
    end
end

%%
pth_out = '../Data_2D/GLM.Ch.main_heatmap';
mkdir2(pth_out);

bml.plot.position_subplots(ax);
file = fullfile(pth_out, 'indiv_n_pooled');
savefigs(file, 'size', [400, 300]);

save(file, 'p_ch', 'p_ch_pooled', 'cond_ch', 'conds', 'cond_rt', 'cond_rt_pooled');
fprintf('Saved to %s.mat\n', file);


%     for dim = 1:n_dim
%         [cond11, ~, dcond11] = ...
%             unique(cond1(:,dim));
%         conds{i_subj, dim} = cond11;
%         dcond{i_subj}(:, dim) = dcond11;
%     end
%     dcond1 = dcond{i_subj};
%     for dim = 1:n_dim
%         cond_ch{i_subj, dim} = accumarray([dcond1, ch1(:,dim)], 1, ...
%             [], @sum);
%     end
% end

