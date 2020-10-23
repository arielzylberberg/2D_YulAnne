clear;
init_path;

%% rename sTr files
SUBJS = SubjOld2New_EXCLUDE.SUBJS;

for parad = {'RT', 'bimanual', 'unimanual', 'VD'}
    for subj = SUBJS.(parad{1})
        file0 = fullfile('../../Data_2D/sTr', ...
            sprintf('%s_%s.mat', parad{1}, subj{1}));
        
        subj1 = SubjOld2New_EXCLUDE.subj_parad2new(subj{1}, parad{1});
        file1 = fullfile('../../Data_2D/sTr', ...
            sprintf('%s_%s.mat', parad{1}, subj1));
        
        if exist(file0, 'file')
            if ~strcmp(file0, file1)
                fprintf('Renaming %s to %s\n', file0, file1);
                movefile(file0, file1);
            else
                fprintf('File name already updated: %s\n', file1);
            end
        else
            fprintf('File absent: %s\n', file0);
        end
    end
end