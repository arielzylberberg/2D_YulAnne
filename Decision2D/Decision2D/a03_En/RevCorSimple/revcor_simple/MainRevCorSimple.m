classdef MainRevCorSimple < matlab.mixin.Copyable
methods
    function tbl = main(Main, sTr, Ss)
        % sTr: a table with the following columns:
        % .i_subj(tr) : index (1, 2, ..)
        % .n_dim_task(tr) : 1 or 2
        % .dim_rel(tr, dim) : boolean
        % .dif_rel(tr, dim_rel) : index (1, 2, ..)
        % .dif_irr(tr, dim_rel) : index (1, 2, ..)
        % .en{tr, dim}(1, fr)
        % .ch(tr, dim) : 1 or 2
        %
        % Ss: array of structs with the following fields:
        % .subj
        % .n_dim_task
        % .dim_rel
        % .dif_irr
        % .align
        % .n_bin_to_pool
        % .en_fds
        
        ress = arrayfun(@(S) Main.compute_lev(sTr, S), Ss);
        ress2 = arrayfun(@(S1, res1) varargin2S(res1, S1), Ss, ress);
        tbl = struct2table(ress2);
    end
    function [Ss, S0, file] = get_Ss(varargin)
        C0 = varargin2C(varargin, {
            'subj', {'S1', 'S2', 'S3'} % num2cell(1:3)
            'n_dim_task', num2cell(1:2)
            'dim_rel', num2cell(1:2)
            'dif_rel', {1:3}
            'dif_irr', {1:3, 4:5}
            'align', {-1, 1}
            'n_bin_to_pool', 12
            'en_fds', {{'nnME', 'mCE'}}
            'lev', {'mean'}
            });
        [file, S0] = get_lev_file(C0);
        Ss = factorizeS(S0);        
    end
    function param_pool = get_param_pool(Main, dim_rel, align)
        if dim_rel == 1
            if align == -1
                param_pool = {
                    'align', -1 % -1: beginning; 0: middle; 1: end
                    'truncate_st', 0 % 12/75 = 0.16 (sec)
                    'truncate_en', 15 % 15/75 = 0.20 (sec)
                    'xlabel', 'Time from onset (s)'
                };
            elseif align == 1
                param_pool = {
                    'align', 1 % -1: beginning; 0: middle; 1: end
                    'truncate_st', 12 % 12/75 = 0.16 (sec)
                    'truncate_en', 0 % 15/75 = 0.20 (sec)
                    'xlabel', 'Time from RT (s)'
                };
            end
        elseif dim_rel == 2
            if align == -1
                param_pool = {
                    'align', -1 % -1: beginning; 0: middle; 1: end
                    'truncate_st', 0 % 12/75 = 0.16 (sec)
                    'truncate_en', 15 % 15/75 = 0.20 (sec)
                    'xlabel', 'Time from onset (s)'
                };
            elseif align == 1
                param_pool = {
                    'align', 1 % -1: beginning; 0: middle; 1: end
                    'truncate_st', 12 % 12/75 = 0.16 (sec)
                    'truncate_en', 0 % 15/75 = 0.20 (sec)
                    'xlabel', 'Time from RT (s)'
                };
            end
        end
    end
    function res1 = compute_lev(Main, sTr, S)
        % sTr: a table with the following fields:
        % .i_subj(tr) : index (1, 2, ..)
        % .n_dim_task(tr) : 1 or 2
        % .dim_rel(tr, dim) : boolean
        % .dif_rel(tr, dim_rel) : index (1, 2, ..)
        % .dif_irr(tr, dim_rel) : index (1, 2, ..)
        % .en{tr, dim}(1, fr)
        % .ch(tr, dim) : 1 or 2
        
        Lev = Lev1D;
        param_pool = varargin2C(Main.get_param_pool(S.dim_rel, S.align), S);

        tr_incl = strcmp(sTr.subj, S.subj) ...
                & (sTr.n_dim_task == S.n_dim_task) ...
                & sTr.dim_rel(:,S.dim_rel) ...
                & ismember(sTr.dif_rel(:,S.dim_rel), S.dif_rel) ...
                & ismember(sTr.dif_irr(:,S.dim_rel), S.dif_irr);
            
        if isfield(S, 't_RDK_dur')
            tr_incl = tr_incl & ismember(sTr.t_RDK_dur, S.t_RDK_dur);
        end

        %%
        [en, wt, t] = ...
            Lev.pool_time(sTr.en(tr_incl, S.dim_rel), ...
                param_pool{:}, ...
                'n_bin_to_pool', S.n_bin_to_pool);
        ch = sTr.ch(tr_incl, S.dim_rel) == 2;

        %%
        if any(tr_incl)
            disp('--');
        end
        [slope, se_slope, bias, se_bias, res, S] = ...
            Lev.slope_by_time( ...
                en, ch, wt, ...
                'cond', sTr.cond(tr_incl, S.dim_rel), ...
                'lev', S.lev);

        nanmean_wt = nanmean(wt) > 0.5;
        t_incl = nanmean_wt > 0.5;
        t1 = t(t_incl);    

        res1 = varargin2S({
            'slope', slope(t_incl)'
            'se_slope', se_slope(t_incl)'
            'bias', bias(t_incl)'
            'se_bias', se_bias(t_incl)'
            'res', res
            'lev', S.lev      
            'nanmean_wt', nanmean_wt(t_incl)
            't', t1
            });
    end
end
end