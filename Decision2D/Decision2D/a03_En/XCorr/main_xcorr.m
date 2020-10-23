%%
clear;
init_path;

%%
L = load('../Data_2D/sTr/combined_2D_RT_sh.mat');
ds = L.dat;
clear L

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
XC = XCorr;
n_subj = 3;
n_dim = 2;
for i_subj = 1:n_subj
    incl = (ds.i_subj == i_subj) & strcmp(ds.parad_short, parad) ...
        & (ds.n_dim_task == 2) ...
        & all(ds.adcond <= 1, 2);
    ds1 = ds(incl, :);
    
    clf;
%     XC.main(ds1);
    XC.bootstrap(ds1, 'align', 'st' ...
        ... , 'max_fr_incl', 45 ...
        ... , 'truncate_st_fr', 15 ...
        );

    pth_out = '../Data_2D/a03_En/main_xcorr';
    S_file = varargin2S({
        'plt', 'xcorr'
        'subj', sprintf('S%d', i_subj)
        });
    file = bml.str.Serializer.convert(S_file);
    savefigs(fullfile(pth_out, file));
end
