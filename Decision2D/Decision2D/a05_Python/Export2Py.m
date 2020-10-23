classdef Export2Py
methods (Static)
    function main_simplify_variables(varargin)
        %%
        S = varargin2S(varargin, {
            'file_in', '../Data_2D/sTr/sTr_eye_all_subj_parad.mat'
            });
        
        %%
        L = load(S.file_in);
        fprintf('Loaded %s\n', S.file_in);
        
        %% Simplify variables for python
        % Make everything either a numeric column vector
        % or a cell column vector of row vectors.
        sTr = struct2table(L);
        
        [~,sTr.i_subj] = ismember(sTr.subj, Data.Consts.subjs_RT);
        
        sTr.dim_rel1 = ismember(sTr.task, {'A', 'H'});
        sTr.dim_rel2 = ismember(sTr.task, {'A', 'V'});        
        sTr.n_dim_task = (sTr.task == 'A') + 1;
        
        [~,sTr.i_parad] = ismember(sTr.parad, {'RT', 'sh'});
        
        sTr.en1 = sTr.nnME;
        sTr.en2 = sTr.mCE;
        
        sTr.cond1 = sTr.condM;
        sTr.cond2 = sTr.condC;
        
        sTr.nnME = [];
        sTr.mCE = [];
        
        sTr.ch1 = sTr.subjM;
        sTr.ch2 = sTr.subjC;

        n_subj = 3;
        n_dim = 2;
        for i_subj = 1:n_subj
            for dim_rel = 1:n_dim
                tr_incl = sTr.i_subj == i_subj;
                [~,~,dif_rel] = unique(abs(sTr.(sprintf('cond%d', dim_rel)) ...
                    (tr_incl, 1)));
                sTr.(sprintf('dif_rel%d', dim_rel))(tr_incl,1) = dif_rel;

                dim_irr = n_dim + 1 - dim_rel;
                [~,~,dif_irr] = unique(abs(sTr.(sprintf('cond%d', dim_irr)) ...
                    (tr_incl, 1)));
                sTr.(sprintf('dif_irr%d', dim_rel))(tr_incl,1) = dif_irr;
            end
        end
        
        xypt = 'xypt';
        xyct = 'xyct';
        z = zeros(1,0);
        for ii = 1:4
            sTr.(['eye_' xypt(ii)]) = cellfun(@(v) v(:,ii)', sTr.eye_xypt, ...
                'UniformOutput', false, 'ErrorHandler', @(varargin) z);
            sTr.(['dot_' xyct(ii)]) = cellfun(@(v) v(:,ii)', sTr.xyct, ...
                'UniformOutput', false, 'ErrorHandler', @(varargin) z);
        end
        
        %% Remove anything but numerical or cell column vector
        for f = sTr.Properties.VariableNames(:)'
            v = sTr.(f{1});
            if ~iscolumn(v)
                fprintf('Removing %s: not a column vector\n', f{1});
                sTr.f({1}) = [];
            elseif ~islogical(v) && ~isnumeric(v) && ~iscell(v)
                fprintf('Removing %s: not a numeric or cell vector\n', ...
                    f{1});
                sTr.(f{1}) = [];
            elseif iscell(v) && any(~cellfun(@isrow, v))
                fprintf('Removing %s: not all row vector\n', f{1});
                sTr.(f{1}) = [];
            elseif iscell(v) && any(cellfun(@ischar, v))
                fprintf('Removing %s: string\n', f{1});
                sTr.(f{1}) = [];
            end                
        end
        
        %%
        L = table2struct(sTr, 'ToScalar', true);

        [pth_in, nam_in] = fileparts(S.file_in);
        file_out = fullfile(pth_in, [nam_in, '_py.mat']);
        save(file_out, '-struct', 'L');
        fprintf('Saved to %s\n', file_out);
    end
end
end