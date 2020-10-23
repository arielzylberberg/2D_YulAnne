classdef EvTimeD1Pdf ...
        < IxnKernel.EvTime.EvTimeD1
    % IxnKernel.EvTime.EvTimeD1Pdf 
    
%% Input - required
properties
    rt_pdf_tr = []; % (trial, fr, ch)
    
    S_dtb = struct; % As saved in dtb.pred.calc_dtb.
end
%% Settings - optional
properties
    % Match the marginal choice proportion of 
    % Td_pred_pdf and RT_pred_pdf to the observed.
    to_match_ch_pred = true; 
end
%% Setup
methods
    function Ev = EvTimeD1Pdf(varargin)
        if nargin > 0
            Ev.init(varargin{:});
        end
    end
end
%% Import
methods
    function import_pdf(Ev, file0, varargin)
        % Import predictions (by default, assuming no Tnd)
        % from what's exported by Fit.D1.BoundedEn.Main.export_pred.
        %
        % Required variables in the file:
        %
        % Td_pred_pdf_tr(fr, tr, ch)
        % RT_pred_pdf_tr(fr, tr, ch)
        % RT_data_pdf_tr(fr, tr, ch)
        % ch(tr, 1)
        % cond(tr, 1)
        % en_rel_mat(tr, fr)
        % rt_fr(tr, 1)
        %
        % Optional variables in the file:
        %
        % th : struct containing the fitted parameters.
        %
        % OPTIONS:
        % 'src', 'Td_pred' % 'Td_pred' | 'RT_data' | 'RT_pred'
        % 'dif_rel_incl', 1 % 'all'
        % 'st_fr', 11
        
        S = varargin2S(varargin, {
            'src', 'Td_pred' % 'Td_pred' | 'RT_data' | 'RT_pred'
            'dif_rel_incl', 1:3 % 'all'
            'st_fr', 11
            });
        
        %%
        fprintf('Importing %s from %s\n', S.src, file0);
        L = load(file0);
        
        %% Import S_file
        try
            [~, nam] = fileparts(file0);
            S_file = bml.str.Serializer.convert(nam);
            Ev.S_file_ = S_file;
        catch
        end
        
        %% Define tr_incl and S_file.
        [~,~,ad_cond] = unique(abs(L.cond));
        n_tr = size(ad_cond, 1);
        tr_incl = true(n_tr, 1);
        if ~isequal(S.dif_rel_incl, 'all')
            tr_incl = tr_incl & bsxEq(ad_cond, S.dif_rel_incl);
        end
        Ev.S_file_.dfr = S.dif_rel_incl;
        Ev.S_file_ = bml.struct.rmfield(Ev.S_file_, {
            'dim_r', 'dif_i', 'acc_i',' dfr', 'dfi', 'aci'
            });
        
        %% Import trial-by-trial PDF
        assert(ismember(S.src, {'Td_pred', 'RT_data', 'RT_pred'}));
        Ev.rt_pdf_tr = L.([S.src, '_pdf_tr'])(:, tr_incl, :);

        %% Import the rest
        Ev.ch = L.ch(tr_incl) == 2;
        Ev.cond = L.cond(tr_incl);
        Ev.ev = L.en_rel_mat(tr_incl, S.st_fr:end);
        
        % Import evidence from drift
%         Ev.ev = L.S_dtb.drift_cond_t(tr_incl, :);
        
        % Import noise_internal or sigmaSq
        Ev.noise_internal = L;

%         Ev.y = L.S_dtb.y;
        
        Ev.td_fr = L.rt_fr(tr_incl);
        Ev.rt_fr = L.rt_fr(tr_incl);
        
        %% Match marginal choice proportion
        if ~isempty(strfind(S.src, 'pred')) && Ev.to_match_ch_pred
            [~,~,d_cond] = unique(L.cond(tr_incl));
            n_cond = max(d_cond);
            p_ch_data = accumarray([d_cond, Ev.ch + 1], 1, [], @sum, 0);
            p_ch_data = bsxfun(@rdivide, p_ch_data, sum(p_ch_data, 2));
            
            for d_cond1 = 1:n_cond
                incl = d_cond == d_cond1;
                p_ch_data1 = p_ch_data(d_cond1, 2);

                ch_pred_tr = permute(sum(Ev.rt_pdf_tr(:, incl, :), 1), ...
                    [2, 3, 1]);
                logit_pred = logit(ch_pred_tr(:,2));
                
                f = @(bias) mean(invLogit(logit_pred + bias)) - p_ch_data1;
                bias1 = fzero(f, [-2, 2]);
                p_pred_adj = invLogit(logit_pred + bias1);
                p_pred_adj2 = [1 - p_pred_adj, p_pred_adj];
                ratio_adj = p_pred_adj2 ./ ch_pred_tr;
                
                Ev.rt_pdf_tr(:, incl, :) = bsxfun(@times, ...
                    Ev.rt_pdf_tr(:, incl, :), ...
                    permute(ratio_adj, [3, 1, 2]));
            end
            
            disp('Matched marginal choice proportion.');
        end        
    end
end
%% Results
methods
    function [ev, ch, S_filt, tr_incl, fr_incl] = get_ev_filtered(Ev, varargin)
        % [ev, ch, S_filt, tr_incl, fr_incl] = get_ev_filtered(Ev, varargin)
        %
        % ch(tr, [n_ch_positive, n_total])
        
        S = varargin2S(varargin, {
            'ch_from_pdf', true
            });
        
        [ev, ~, S_filt, tr_incl, fr_incl] = ...
            get_ev_filtered@IxnKernel.EvTime.EvTimeD1(Ev, varargin{:});

        if S.ch_from_pdf
            % Sum across all time, not just included choice
            fr_incl_all = repmat({':'}, size(fr_incl));
            rt_pdf_incl = Ev.get_rt_pdf_incl(Ev.rt_pdf_tr, ...
                tr_incl, fr_incl_all);

            ch = permute(nansum(rt_pdf_incl, 2), [1, 3, 2]);
            ch_sum = sum(ch, 2); % sum across choices
            ch(:,1) = ch(:,2);
            ch(:,2) = ch_sum;
        else
        	assert(iscolumn(Ev.ch));
            ch = Ev.ch == 1;
%             ch = [1 - Ev.ch, ones(size(Ev.ch))];
%             ch = ch(tr_incl, :);
        end        
    end
    function rt_pdf_incl = get_rt_pdf_incl(~, rt_pdf_tr, tr_incl, fr_incl)
        % td_pdf_tr(fr, tr, ch)
        % tr_incl(tr,1) : logical
        % fr_incl{tr_incl(ii),1)(1,jj) : frame
        %
        % rt_pdf_incl(tr_incl, fr_incl, ch)
        %
        % If some fr_incl{tr} is shorter, the rest is padded with NaN.
        
        if islogical(tr_incl)
            tr_incls = find(tr_incl);
            n_tr_incl = nnz(tr_incl);
        else
            tr_incls = tr_incl;
            n_tr_incl = length(tr_incl);
        end
        
        n_ch = size(rt_pdf_tr, 3);
        rt_pdf_incl_cell = cell(n_tr_incl, n_ch);
        for ch = n_ch:-1:1
            for ii = 1:n_tr_incl
                tr = tr_incls(ii);
                rt_pdf_incl_cell{ii, ch} = rt_pdf_tr(fr_incl{ii}, tr, ch)';
            end
        end
        rt_pdf_incl = cell2mat2(rt_pdf_incl_cell(:));
        rt_pdf_incl = permute( ...
            reshape(rt_pdf_incl, n_tr_incl, n_ch, []), ...
            [1, 3, 2]);
    end
end
%% Tests
methods
    function demo_logistic_weight(Ev)
        %%
        n_tr = 10;
        X = (mod(1:n_tr, 2) == 0)';
        
    end
end
end