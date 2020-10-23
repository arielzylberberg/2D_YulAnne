classdef LevSegment < EnIxn.Common.DataFilterEn
% Leverage of a segment of a given length.
%
% EnIxn.GLM.LevSegment
% Copied from EnIxn.GLM.MainGLM
    
%% Settings
properties
    lev_kind = 'beta'; % 'beta'|'raw'
    
    seg_loc = 'st' % 'st'|'md'|'en' % segment location - start, middle, end
    seg_len_ms = 500; % segment length in seconds
    
%     % seg_offset_ms - TODO - to exclude the Tnd just before RT.
%     % : segment offset from the aligned point.
%     seg_offset_ms = 200; 
    
    res_props = {'est', 'ci', 'se'};
end
%% Results
properties
    est
    ci
    se
    
    est_full
    ci_full
    se_full
end
%% Init
methods
    function W = LevSegment(varargin)
        W.dif_rel_incl = 1;

        if nargin > 0
            W.init(varargin{:});
        end
    end
end
%% Main
methods
    function main(W)
        W.calculate;
    end
end
%% Calculate
methods
    function calculate(W)
        %% Load results if existing
        [loaded, file] = W.load_if_existing;
        if loaded
            return;
        end
        
        %%
        en = W.get_ens_cell;
        en_rel_cell0 = en{W.dim_rel_W};

        %% Get first/middle/last part of the en
        n_fr_seg = round(W.seg_len_ms / 1e3 / W.dt);
        n_fr_tr = cellfun(@numel, en_rel_cell0);
        n_tr = size(en_rel_cell0, 1);
        switch W.seg_loc
            case 'st'
                fr_st = ones(n_tr, 1);
                
            case 'md'
                fr_st = max(round((n_fr_tr + 1 - n_fr_seg) / 2), 1);
                
            case 'en'
                fr_st = max(n_fr_tr - n_fr_seg, 1);
                
            otherwise
                error('Unknown seg_loc = %s\n', W.seg_loc);
        end     
        fr_en = min(n_fr_tr, fr_st + n_fr_seg - 1);
        
        en_rel_full = cellfun(@nanmean, en_rel_cell0);
        en_rel = arrayfun(@(c, st, en) nanmean(c{1}(st:en)), ...
            en_rel_cell0, fr_st, fr_en);
        
        %%
        ch = W.Data.ch == 2;
        ch_rel = ch(:, W.dim_rel_W);
        
        switch W.lev_kind
            case 'beta'
                roni = bml.stat.ind_cols(W.Data.cond(:,W.dim_rel_W), 0);
                
                x = standardize([en_rel_full, roni]);
                res = glmwrap(x, ch_rel, 'binomial');
                est_full = res.b(2);
                se_full = res.se(2);
                
                x = standardize([en_rel, roni]);
                res = glmwrap(x, ch_rel, 'binomial');
                est = res.b(2);
                se = res.se(2);
                
                est = est - est0;
                
                
%             case 'logit_for_ch'
%                 roni = bml.stat.ind_cols(W.Data.cond(:,W.dim_rel_W), 0);
%                 
%                 res = glmwrap([en_rel, roni], ch_rel, 'binomial');
%                 y = bml.stat.y_for_ch(en_rel, ch_rel, res.b(2));
%                 est = nanmean(y);
%                 se = sqrt((res.se(2) .^ 2) + nansem(y).^2);
                
%             case 'raw'
%                 [~, ~, dcond] = unique(W.Data.cond(:,W.dim_rel_W));
%                 en_rel1 = en_rel;
%                 ndcond = max(dcond);
%                 if ndcond > 1
%                     for dcond1 = 1:ndcond
%                         incl = dcond == dcond1;
%                         mdcond = nanmean(en_rel(incl,:));
%                         en_rel1(incl,:) = bsxfun(@minus, ...
%                             en_rel1(incl,:), mdcond);
%                     end
%                 end
%                 
%                 y = bml.stat.y_for_ch(en_rel1, ch_rel);
%                 est = nanmean(y);
%                 se = nansem(y);
                
            otherwise
                error('Unknown lev_kind=%s\n', W.lev_kind);
        end
        
        ci = [est - se, est + se];
        ci_full = [est_full - se_full, est_full + se_full];
        
        %%
        W.est = est;
        W.ci = ci;
        W.se = se;
        
        W.est_full = est_full;
        W.ci_full = ci_full;
        W.se_full = se_full;
        
        L = struct;
        L = copyprops(L, W, 'props', W.res_props);
        L = copyFields(L, W.S0_file);
        mkdir2(fileparts(file));
        save(file, '-struct', 'L');
        fprintf('Saved results to %s.mat\n', file);
    end
    function [loaded, file] = load_if_existing(W)
        file = W.get_file;
        does_exist = exist([file, '.mat'], 'file');
        if does_exist && W.skip_existing_mat
            L = load(file);
            fprintf('Loaded results from %s.mat\n', file);
            copyprops(W, L, 'props', W.res_props);
            loaded = true;
        else
            if does_exist
                fprintf('Results exist but not skipping!\n');
            else
                fprintf('Results don''t exist: %s.mat\n', file);
            end
            loaded = false;
        end
    end
end
%% File
methods
    function v = get_file_fields0(W)
        v = union_general({
            'lev_kind', 'lev'
            'seg_loc', 'slc'
            'seg_len_ms', 'sln'
            }, W.get_file_fields0@EnIxn.Common.DataFilterEn, ...
            'stable', 'rows');
    end
end
end