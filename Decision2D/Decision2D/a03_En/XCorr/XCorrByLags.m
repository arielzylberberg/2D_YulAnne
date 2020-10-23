classdef XCorrByLags
methods
    function [S, C] = parse_args(XC, varargin)
        S = varargin2S(varargin, {
            'truncate_st_fr', 15
            'truncate_en_fr', 15
            'max_fr_incl', 75
            'exclude_short_trs', true
            'align', 1
            'align2', []
            ...
            'mask_fr_st', 0
            'smooth_fr', 0 % 5
            ...
            'n_shuffle', 200
            'to_shuffle' true
            ...
            'refresh_rate', 75
            'to_plot', true
            ...
%             'dif1', 1:2 % done in main_xcorr_by_lags
%             'dif2', 1:2
            ...
            'fold', 'sym' % 'sym'|'asym'|'none'
            'remove_mean', 'all' % 'cond' % 'cond', 'cond_t', 'none'
            });
        if isempty(S.align2)
            S.align2 = S.align;
        end
        C = S2C(S);        
    end
    function main(XC, ds1, varargin)
        [S, C0] = XC.parse_args(varargin{:});
        
        C = varargin2C({
            'to_shuffle', false
            }, C0);
        [en, ch, dcond, rt] = deal(ds1.en, ds1.ch, ds1.dcond, ds1.RT);
        [en, tr_incl] = XC.preprocess_en(en, rt, C{:});
        ch = ch(tr_incl, :);
        dcond = dcond(tr_incl, :);
        
        fprintf('#tr_incl: %d, P(tr_incl): %1.2f\n', ...
            nnz(tr_incl), nanmean(tr_incl));
        
        xcs0 = XC.calc_xcorr(en, ch, dcond, C{:});
        m_xcs0 = nanmean(xcs0, 3);
        e_xcs0 = nansem(xcs0, 3);
        
        %%
        if S.to_shuffle
            n_par = 4;
            m_xcss_par = cell(n_par, 1);
%             m_xcss = cell(S.n_shuffle, 1);
            parfor i_par = 1:n_par
                n_shuf_per_par = S.n_shuffle / n_par;
                shufs = round(n_shuf_per_par * (i_par - 1) + 1): ...
                    round(n_shuf_per_par * i_par);
                n_shuf1 = numel(shufs);
                
                m_xcss1 = cell(n_shuf1, 1);
                for i_shuf1 = 1:n_shuf1
                    i_shuffle = shufs(i_shuf1);
                    rng(i_shuffle);
                    en1 = XC.shuffle(en, ch, dcond, C{:});
                    xcs1 = XC.calc_xcorr(en1, ch, dcond, C0{:});
                    m_xcss1{i_shuf1} = nanmean(xcs1, 3);
%                     m_xcss{i_shuffle} = nanmean(xcs1, 3);
                end
                m_xcss_par{i_par} = m_xcss1;
            end
            m_xcss = cat(1, m_xcss_par{:});
            m_xcss = cat(3, m_xcss{:});
        else
            m_xcss = zeros([sizes(m_xcs0, [1, 2]), 1]);
        end
        mm_xcss = nanmean(m_xcss, 3);
        se_xcss = nansem(m_xcss, 3);
        
        m_xcs = m_xcs0 - mm_xcss;
        
        %%
        [m_by_dfr0, e_by_dfr0, dt] = XC.marginalize(m_xcs0, varargin{:});
        m1 = zeros(length(dt), S.n_shuffle);
        for i_shuffle = 1:S.n_shuffle
            m1(:,i_shuffle) ...
                = XC.marginalize(m_xcss(:,:,i_shuffle), varargin{:});
        end
        
        m1 = smooth_gauss(m1, S.smooth_fr);
        m_by_dfr0 = smooth_gauss(m_by_dfr0, S.smooth_fr);
        
        switch S.fold
            case 'asym'
                m1 = (m1 - flip(m1, 1)) / 2;
                m_by_dfr0 = (m_by_dfr0 - flip(m_by_dfr0)) / 2;
            case 'sym'
                m1 = (m1 + flip(m1, 1)) / 2;
                m_by_dfr0 = (m_by_dfr0 + flip(m_by_dfr0)) / 2;
            case 'power'
                m1 = sqrt((m1 - flip(m1, 1)).^2 + (m1 + flip(m1, 1)).^2);
                m_by_dfr0 = sqrt( ...
                    (m_by_dfr0 - flip(m_by_dfr0)).^2 ...
                    + (m_by_dfr0 + flip(m_by_dfr0)).^2);
            case 'none'
                % do nothing
            otherwise
                error('Unsupported fold=%s', S.fold);
        end        
        m_shuf = median(m1, 2);
        if strcmp(S.fold, 'power')
            e_shuf = prctile(m1, [0, 95], 2) - m_shuf;
        else
            e_shuf = prctile(m1, [2.5, 97.5], 2) - m_shuf;
        end
        
        %%
        if S.to_plot
%             plot(dt, m1(:,randi(end)), '-', 'Color', 0.5 + [0,0,0]);
%             hold on;

            % Errorbar: 95% CI across shuffles
            errorbarShade(dt, m_shuf, e_shuf, 'k--');
            hold on;

            % Errorbar: 95% CI across ???
            errorbarShade(dt, m_by_dfr0, e_by_dfr0, 'r-');
%             plot(dt, m_by_dfr0, 'r-', 'LineWidth', 2)
            hold off;
            
            crossLine('h', 0, 'k--');
            crossLine('v', 0, 'k--');
            bml.plot.beautify;
            
            switch S.fold
                case {'sym', 'asym', 'power'}
                    xlim([0, 1.05] * S.max_fr_incl / 75);            
                case 'none'
                    xlim([-1.05, 1.05] * S.max_fr_incl / 75);            
            end
        end
        
%         %%
%         if S.to_plot
%             XC.imagesc(m_xcs, varargin{:});
%         end
    end
    function [en, tr_incl] = preprocess_en(XC, en, rt, varargin)
        [S, C] = XC.parse_args(varargin{:});

        n_dim = 2;
        n_tr = size(en, 1);
        n_fr0 = min(min( ...
            round(rt * S.refresh_rate), ...
            cellfun(@numel, en)), ...
            [], 2);
        if S.exclude_short_trs
            tr_incl = n_fr0 >= (S.truncate_st_fr + S.truncate_en_fr ...
                + S.max_fr_incl);
        else
            tr_incl = true(n_tr, 1);
        end
        
%         if ~isempty(S.dif1)
%             tr_incl = tr_incl & ismember(dcond(:,1), S.dif1);
%         end
%         if ~isempty(S.dif2)
%             tr_incl = tr_incl & ismember(dcond(:,2), S.dif2);
%         end        
        
        align = [S.align, S.align2];
        
        for dim_rel = 1:n_dim
            en1 = en(:,dim_rel);
            
%             en1 = cellfun(@(v) smooth_gauss(v, S.smooth_fr), en1, ...
%                 'UniformOutput', false);
%             
            en1 = arrayfun(@(v, n_fr1) v{1}(1:n_fr1), ...
                en1, n_fr0, 'UniformOutput', false);
            C = varargin2C({
                'align', align(dim_rel)
                }, S);
            en1 = align_time(en1, C{:});

            for tr = 1:n_tr
                en11 = en1{tr};
                n_fr11 = numel(en11);
                if isempty(en11) || ...
                        (S.exclude_short_trs ...
                            && n_fr11 < S.max_fr_incl)
                    en1{tr} = nan(1, S.max_fr_incl);
                    continue;
                end
                en11 = en11 - nanmean(en11); % TODO: see if this is problematic
                en11 = en11(1:min(S.max_fr_incl, n_fr11));
                en1{tr} = [en11, ...
                    nan(1, max(0, S.max_fr_incl - length(en11)))];
            end
            en(:,dim_rel) = en1;
        end        
        en = en(tr_incl, :);
    end
    function xcs = calc_xcorr(XC, en, ch, dcond, varargin)
        % en{tr, dim}
        % ch(tr, dim) = 1 or 2
        % dcond(tr, dim)
        %
        % xcs(t_dim1, t_dim2, tr)
        
        [S, C] = XC.parse_args(varargin{:});

        % remove mean within each (dcond, t)
        for dim = 1:2
            switch S.remove_mean
                case 'all'
                    en1 = en(:, dim);
                    en1 = cell2mat2(en1);
                    en1 = en1 - nanmean(en1(:)); 
                    en(:, dim) = row2cell(en1);
            end
            
            [~,~,i_dcond] = unique(dcond(:,dim));
            n_dcond1 = max(i_dcond);
            for i_dcond1 = 1:n_dcond1
                tr_incl = i_dcond == i_dcond1;
                en1 = en(tr_incl, dim);
                en1 = cell2mat2(en1);
                switch S.remove_mean
                    case 'cond_t'
                        en1 = en1 - nanmean(en1, 1);
                    case 'cond'
                        en1 = en1 - nanmean(en1(:)); 
                end
                en(tr_incl, dim) = row2cell(en1);
            end
        end
        
        % calculate en for ch
        en_for_ch = arrayfun(@(en1, ch1) en1{1} .* sign(ch1 - 1.5), ...
            en, ch, 'UniformOutput', false);
        
        %%
        xcs = cellfun(@(en1, en2) en1(:) .* en2(:)', ...
            en_for_ch(:,1), en_for_ch(:,2), ...
            'UniformOutput', false);
        xcs = cat(3, xcs{:}); 
    end
    function en = shuffle(XC, en, ch, dcond, varargin)
        [S, C] = XC.parse_args(varargin{:});
        
        n_dim = 2;
        for dim = 1:n_dim
            [~,~,group] = unique([dcond(:,dim), ch(:,dim)], 'rows');
            tr1 = bml.stat.shuffle_wi_group(group);
            en(:,dim) = en(tr1,dim);
        end
    end    
    function [m, e, dt] = marginalize(XC, xcs, varargin)
        S = XC.parse_args(varargin{:});
        
        %%
        n_fr1 = size(xcs, 1);
        assert(size(xcs, 2) == n_fr1);
        
        [frs1, frs2] = ndgrid(1:n_fr1, 1:n_fr1);
        
        if S.mask_fr_st
            incl0 = (frs1 > S.mask_fr_st) & (frs2 > S.mask_fr_st);
        else
            incl0 = true(size(frs1));
        end
        
        dfr21 = frs2 - frs1;
        
        %%
        dfrs = -(n_fr1 - 1):(n_fr1 - 1);
        dt = dfrs ./ S.refresh_rate;
        n_dfr = numel(dfrs);
        m = zeros(n_dfr, 1);
        e = zeros(n_dfr, 2);
        n_rep = size(xcs, 3);
        for i_dfr = 1:n_dfr
            dfr1 = dfrs(i_dfr);
            incl = repmat((dfr21 == dfr1) & incl0, [1, n_rep]);
            m(i_dfr) = nanmean(vVec(xcs(incl)));
            e(i_dfr, :) = nansem(vVec(xcs(incl))) .* [-1, 1];
%             m(i_dfr) = median(vVec(xcs(incl)));
%             e(i_dfr, :) = prctile(vVec(xcs(incl)), [2.5, 97.5]);            
        end
%         e = e - m;
    end
end
%% Plotting
methods
    function imagesc(XC, m_xcs, varargin)
        S = XC.parse_args(varargin{:});
        
        t = (0:(S.max_fr_incl - 1)) / S.refresh_rate;
        imagesc(t, t, m_xcs);
        axis equal
        axis tight
        axis xy
        bml.plot.beautify;
        h = crossLine('NE', 0, 'k:');
        uistack(h, 'top');

        switch S.align
            case 1
                set(gca, 'XDir', 'normal');
                txt = 'from stim onset (s)';
            case -1
                set(gca, 'XDir', 'reverse');
                txt = 'from RT (s)';
        end
        xlabel({'t_{motion} ', txt});
        ylabel({'t_{color} ', txt});       
        
        switch S.align2
            case 1
                set(gca, 'YDir', 'normal');
                txt = 'from stim onset (s)';
            case -1
                set(gca, 'YDir', 'reverse');
                txt = 'from RT (s)';
        end
        ylabel({'t_{color} ', txt});        
    end
end
end