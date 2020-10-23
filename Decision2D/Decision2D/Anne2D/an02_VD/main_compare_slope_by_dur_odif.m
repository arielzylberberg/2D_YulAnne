%%
% L = load('../Data_2D/sTr/combined_2D_RT_sh_VD_unibimanual.mat');

%%
incl = (L.id_parad == find(strcmp(L.parads, 'VD'))) ...
    & (L.n_dim_task == 2);
incl = incl & (L.id_subj == min(L.id_subj(incl)));

% en = L.en(incl, :, :);
% for dim = 1:2
%     subplot(2, 1, dim);
%     plot(squeeze(en(1, :, dim)));
% end

ch = L.ch(incl, :);
cond = L.cond(incl, :);
dur = L.t_RDK_dur(incl, :);

%%
n_dim = 2;
aconds = cell(1, n_dim);
clear adcond
for dim = n_dim:-1:1
    [aconds{dim}, ~, adcond(:,dim)] = unique(abs(cond(:,dim)));
end
[durs, ~, ix_dur] = unique(dur);
accu = sign(cond) == sign(ch - 0.5);

%%
clear b
adcond_incl = {[1, 2], 3};
for i_dur = 1:numel(durs)
    for i_dif = 1:numel(adcond_incl)
        for dim = 1:2
            odim = 3 - dim;
            incl1 = (ix_dur == i_dur) ...
                & ismember(adcond(:,odim), adcond_incl{i_dif}) ...
                & accu(:,odim);

            b1 = glmfit(cond(incl1, dim), ch(incl1, dim), 'binomial');
            b(i_dur, i_dif, dim) = b1(2);
        end
    end
end

%%
for dim = 1:2
    subplot(2, 1, dim);
    plot(b(:,:,dim));
    y_lim = ylim;
    ylim([0, y_lim(2)]);
end