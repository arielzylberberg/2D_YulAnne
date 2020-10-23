classdef EyeData < matlab.mixin.Copyable
methods
    function main_combine_eye_data(ED)
        %%
        subjs = Data.Consts.subjs_RT;
        n_subj = 3;
        for i_subj = 1:n_subj
            subj = subjs{i_subj};
            ED.combine_eye_data(subj);
        end
    end
    function combine_eye_data(ED, subj)
        file_sTr = fullfile('../Data_2D/sTr', sprintf('RT_%s.mat', subj));
        L = load(file_sTr);
        fprintf('Loaded %s\n', file_sTr);
        
        %%
        trial_files = strrep(L.dat.trialFile, ...
            '/Users/yul/Dropbox/CodeNData_2D/Data_2D/Shadlen/', ...
            '../Data_2D/');
        n_tr = size(L.dat, 1);
        eye_xypt = cell(n_tr, 1);
        t_RDK_on_absSec = nan(n_tr, 1);
        t_RDK_off_absSec = nan(n_tr, 1);
        
        %%
        for tr = 1:n_tr
            %%
            L1 = load(trial_files{tr});
            t_RDK_on_absSec(tr) = L1.RDKCol.absSec('on');            
            t_RDK_off_absSec(tr) = L1.RDKCol.absSec('off');

            xypt = [
                L1.Eye.v('xyDeg')
                L1.Eye.v('pupil')
                L1.Eye.t('xyDeg') - t_RDK_on_absSec(tr)
                ]';
            eye_xypt{tr} = xypt;
            
            if mod(tr, 10) == 0
                fprintf('.');
            end
            if mod(tr, 100) == 0
                fprintf('%d/%d trials processed\n', tr, n_tr);
            end
        end
        
        %%
        L.dat.eye_xypt = eye_xypt;
        L.dat.t_RDK_on_absSec = t_RDK_on_absSec(:);
        L.dat.t_RDK_off_absSec = t_RDK_off_absSec(:);
        
        %%
        file_out = fullfile('../Data_2D/sTr', sprintf('RT_%s_eye.mat', subj));
        fprintf('Saving to %s ...', file_out);
        save(file_out, '-v7', '-struct', 'L');
        fprintf('Done.\n');
    end        
end
methods (Static)
    function main_combine_across_subj_parad
        %% First concatenate datasets
        parads = {'RT'};
        subjs = {'S1', 'S2', 'S3'};
        pth = '../Data_2D/sTr';
        dat = dataset;
        n_tr = 0;
        
        for parad = parads
            for subj = subjs
                file1 = fullfile(pth, ...
                    sprintf('%s_%s_eye.mat', parad{1}, subj{1}));
                L1 = load(file1);
                fprintf('Loaded %s\n', file1);
                
                n_tr1 = size(L1.dat, 1);
                L1.dat.subj = repmat(subj, [n_tr1, 1]);
                L1.dat.parad = repmat(parad, [n_tr1, 1]);
                
                dat = bml.ds.ds_set(dat, n_tr + (1:n_tr1), L1.dat);
                n_tr = n_tr + n_tr1;
                fprintf('Added %d rows\n', n_tr1);
            end
        end
        
        %% Copy only relevant fields to L
        L = struct;
        fs = {'parad', 'subj', 'task', 'condC', 'condM', ...
            'gap', 't_RDK_dur', 'RT', 'i_Tr', 'i_all_Tr', 'i_Run', ...
            'corrC', 'corrM', 'subjC', 'subjM', 'score', ...
            'mCE', 'mME', 'nnME', 'xyct', 'eye_xypt'};
        L = copyFields(L, dat, fs);
        file_out = '../Data_2D/sTr/sTr_eye_all_subj_parad.mat';
        save(file_out, '-struct', 'L');
        fprintf('Saved to %s\n', file_out);
        
        %%
        info = dir(file_out);
        fprintf('File size: %g MB\n', info.bytes / 1e6);
    end
end
end