classdef XCorr
methods
    function bootstrap(XC, ds1, varargin)
        %%
        S = varargin2S(varargin, {
            'n_boot', 100
            'seed', 1
            });
        C = varargin2C({
            'to_plot', false
            }, S);
        
        rng(S.seed);
        [~,~,group] = unique(ds1.cond, 'rows');
        for i_boot = S.n_boot:-1:1
            ix_tr(:, i_boot) = bml.stat.randsample_group(group);
        end
        
        m_xc0 = cell(1, S.n_boot);
        tic;
        for i_boot = 1:S.n_boot
            [m_xc0{i_boot}, res] = XC.main(ds1(ix_tr(:, i_boot),:), C{:});
        end
        toc;
        m_xc0 = cat(1, m_xc0{:});
        
        %%
        m_xc = mean(m_xc0);
        sem_xc = std(m_xc0);
        
        errorbarShade(res.t, m_xc, sem_xc);
        crossLine('h', 0, 'k--');
        crossLine('v', 0, 'k--');
        bml.plot.beautify;
    end
    function [m_xc, res, S] = main(XC, ds1, varargin)
        
        S = varargin2S(varargin, {
            'truncate_st_fr', 15
            'truncate_en_fr', 15
            'max_fr_incl', 75
            'align', 'st'
            'refresh_rate', 75
            'dfr_max', 100
            ...
            'to_shuffle', false
            'seed_shuffle', 1
            ...
            'to_plot', true
            });
        
        %% Preprocess
        n_cond = max(ds1.dcond);
        
        n_dim = 2;
        n_tr = size(ds1, 1);
        en = ds1.en;
        ch = ds1.ch;
        n_fr0 = min( ...
            round(ds1.RT * S.refresh_rate), ...
            cellfun(@numel, en));
        en_for_ch = cell(size(en));
        for dim_rel = 1:n_dim
            for tr = 1:n_tr
                en1 = en{tr, dim_rel};
                ch1 = ch(tr, dim_rel);
                
                switch S.align
                    case 'st'
                        en1 = en1((S.truncate_st_fr + 1) ...
                                :min(S.max_fr_incl, ...
                                     n_fr0(tr, dim_rel) ...
                                     - S.truncate_en_fr));
                    case 'en'
                        en1 = en1(max(S.truncate_st_fr + 1, ...
                                        n_fr0(tr, dim_rel) ...
                                        - S.max_fr_incl) ...
                                :(n_fr0(tr, dim_rel) ...
                                     - S.truncate_en_fr));
                end
                if ~isempty(en1)
                    en1 = en1 - mean(en1);
                end
                en1 = en1 .* sign(ch1 - 1.5);
                
                en_for_ch{tr, dim_rel} = en1;                    
            end
        end
        n_fr = cellfun(@numel, en_for_ch(:,1));
        
        dfrs = -S.dfr_max:S.dfr_max;
        n_dfr = numel(dfrs);
        n_fr_in = max(bsxfun(@minus, n_fr, abs(dfrs)), 0);
        
        %%
        xc = cell(n_tr, 1);
        xc = XC.calc_xcorr(en_for_ch, S.dfr_max);
%         for cond1 = 1:n_cond(1)
%             for cond2 = 1:n_cond(2)
%                 tr_incl = (ds1.dcond(:,1) == cond1) ...
%                         & (ds1.dcond(:,2) == cond2);
%                 xc0(tr_incl) = XC.calc_xcorr(en_for_ch(tr_incl, :), ...
%                     S.dfr_max);
%             end
%         end
        xc = cell2mat2(xc);
        m_xc = nanmean(xc);
        t = dfrs ./ S.refresh_rate;
        
        res = packStruct(xc, t);

        if S.to_plot
            clf;
            errorbar(t, nanmean(xc), nansem(xc));
            hold on;
            errorbar(t, nanwmean(xc, n_fr_in), nanwsem(xc, n_fr_in));
            crossLine('h', 0, 'k--');
            crossLine('v', 0, 'k--');
            hold off;
            bml.plot.beautify;
        end
         
    end
    function xc = calc_xcorr(XC, en_for_ch, dfr_max)
        n_tr = size(en_for_ch, 1);
        n_fr = cellfun(@numel, en_for_ch(:,1));
        xc = cell(n_tr, 1);
        for tr = 1:n_tr
            en1 = en_for_ch{tr, 1};
            en2 = en_for_ch{tr, 2};
            
            en1 = en1 - nanmean(en1);
            en2 = en2 - nanmean(en2);
            
            if isempty(en1)
                xc1 = [];
            else
                [xc1, lags1] = xcorr(en1, en2, dfr_max);
                try
                    xc1(abs(lags1) >= n_fr(tr)) = nan;
                catch err
                    warning(err_msg(err));
                end
            end
            
            xc{tr} = xc1;
        end
    end
end
end