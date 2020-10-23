L = load('../Data_2D/sTr/combined_2D_RT_sh_VD_unibimanual.mat');

%%
incl = (L.id_parad == find(strcmp(L.parads, 'VD'))) ...
    & (L.n_dim_task == 2);
incl = incl & (L.id_subj == min(L.id_subj(incl)));
en = L.en(incl, :, :);

for dim = 1:2
    subplot(2, 1, dim);
    plot(squeeze(en(1, :, dim)));
end
