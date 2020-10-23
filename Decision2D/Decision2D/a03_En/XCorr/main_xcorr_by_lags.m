% %%
% clear;
% init_path;
% 
% L = load('../Data_2D/sTr/combined_2D_RT_sh.mat');
% ds = L.dat;
% clear L

%%
parad = 'RT';
ds.dim_rel = bsxfun(@eq, 'HV', ds.task);
ds.n_dim_task = (ds.task == 'A') + 1;
ds.dim_rel(ds.task == 'A', :) = true;
n_dim = 2;
ds.cond = [ds.condM, ds.condC];
ds.ch = [ds.subjM, ds.subjC];
if ismember('nnME', ds.Properties.VarNames)
    ds.en = [ds.nnME, ds.mCE];
    ds.nnME = [];
    ds.mCE = [];
end
n_subj = max(ds.i_subj);
for i_subj = 1:n_subj
    incl = ds.i_subj == i_subj;
    for dim = n_dim:-1:1
        [~, ~, ds.adcond(incl,dim)] = unique(abs(ds.cond(incl, dim)));
        [~, ~, ds.dcond(incl,dim)] = unique(ds.cond(incl, dim));
    end
end

%%
pth_out = '../Data_2D/a03_En/main_xcorr_by_lags';

%%
XC = XCorrByLags;
n_subj = 3;
n_dim = 2;
max_fr_incl = 75;
truncate_st_fr = 15;
truncate_en_fr = 15;
exclude_short_trs = false;
% fold = 'sym';
smooth_fr = 2;
remove_mean = 'all'; % 'cond'|'cond_t'|'all'|'none'
n_shuffle = 200;

aligns = [-1, 1];
label_align = {'onset', 'offset'};
for fold1 = {'none'} % {'power'} % {'sym', 'asym'} % {'none', 'sym', 'asym'} % ,
    fold = fold1{1};
    for i_subj = 1:n_subj
        incl = (ds.i_subj == i_subj) & strcmp(ds.parad_short, parad) ...
            & (ds.n_dim_task == 2) ...
            & all(ds.adcond <= 2, 2);
        ds1 = ds(incl, :);
        
        clf;
        ax = subplotRCs(2,2);        
        
        for i_align1 = 1:2
            align1 = aligns(i_align1);
            for i_align2 = 1:2
                align2 = aligns(i_align2);
                %%
                axes(ax(i_align1, i_align2));
                fprintf('=== subj %d, align %d, %d\n', ...
                    i_subj, i_align1, i_align2);
    
                XC.main(ds1, 'align', align1, 'align2', align2, ...
                    'truncate_en_fr', 0, ...
                    'max_fr_incl', max_fr_incl, ...
                    'exclude_short_trs', exclude_short_trs, ...
                    'smooth_fr', smooth_fr, ...
                    'fold', fold, ...
                    'remove_mean', remove_mean, ...
                    'n_shuffle', n_shuffle);

                str = sprintf('S%d, M:%s, C:%s, fold:%s', ...
                    i_subj, ...
                    label_align{i_align1}, ...
                    label_align{i_align2}, fold);
                title(str);

                if i_align1 == 2 && i_align2 == 1
                    xlabel('t_C - t_M (s)');
                else
                    set(gca, 'XTickLabel', '');
                end
                set(gca, 'YTickLabel', '');
            end
        end
        sameAxes(ax);
        bml.plot.position_subplots(ax, ...
            'margin_left', 0.01, ...
            'margin_right', 0.01, ...
            'margin_top', 0.05, ...
            'btw_col', 0.01, ...
            'btw_row', 0.075, ...
            'margin_bottom', 0.12);
        
        S_file = varargin2S({
            'plt', 'xcorr_by_lags'
            'sbj', i_subj
            'maxfr', max_fr_incl
            'trc_st', truncate_st_fr
            'trc_en', truncate_en_fr
            'excl_short', exclude_short_trs
            'fold', fold
            'sm', smooth_fr
            'rm_mean', remove_mean
            'nshuf', n_shuffle
            });
        nam = bml.str.Serializer.convert(S_file);
        file = fullfile(pth_out, nam);
        savefigs(file);
    end
end
