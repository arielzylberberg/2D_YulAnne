%%
clear
init_path;

%%
file = '../../Data_2D/sTR/combined_2D_RT_sh_VD_unibimanual.mat';
L = load(file);

%% Recover subj and parad names as cell of str
L.subj = L.subjs(L.id_subj);
L.parad = L.parads(L.id_parad);

%% Replace subj and parad names
SUBJS = SubjOld2New_EXCLUDE.SUBJS;

for parad = {'RT', 'bimanual', 'unimanual', 'VD'}
    for subj = SUBJS.(parad{1})
        subj1 = SubjOld2New_EXCLUDE.subj_parad2new(subj{1}, parad{1});

        if ismember(parad, {'bimanual', 'unimanual'})
            parad0 = 'unibimanual';
        else
            parad0 = parad{1};
        end
        incl = strcmp(L.subj, subj{1}) & strcmp(L.parad, parad0);
        L.subj(incl) = {subj1};
        
        fprintf('%d %s (%s) -> %s\n', ...
            nnz(incl), subj{1}, parad{1}, subj1);
    end
end

%% Replace subj and parad names as id_* + *s
[L.subjs, ~, L.id_subj] = unique(L.subj);

%% Remove subj and parad, which takes up lots of space
L = rmfield(L, {'subj', 'parad'});

%% Save again
save(file, '-v7', '-struct', 'L');
fprintf('Saved to %s\n', file);