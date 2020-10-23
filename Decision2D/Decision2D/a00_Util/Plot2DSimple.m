classdef Plot2DSimple
properties
end
methods
    function plot_ch_by_irr_dif(~, ch, cond_rel, cond_irr, varargin)
        S = varargin2S(varargin, {
            'plot_args', {}
            });
        
        %%
        incl = ~isnan(ch);
        ch = ch(incl);
        cond_rel = cond_rel(incl);
        cond_irr = cond_irr(incl);
        
        [difs_irr,~,dif_irr] = unique(abs(cond_irr));
        n_dif = length(difs_irr);
        colors = row2cell(hsv2rev(n_dif));
        
        [conds_rel,~,i_cond_rel] = unique(cond_rel);
        n_conds_rel = length(conds_rel);
        
        n_ch = accumarray([i_cond_rel, dif_irr, ch], 1, ...
            [n_conds_rel, n_dif, 2], @sum, 0);
        p_ch = n_ch(:,:,2) ./ sum(n_ch, 3);
        
        for i_dif = 1:n_dif
            plot_args1 = varargin2plot(S.plot_args, {
                '-', ...
                'Color', colors{i_dif}, ...
                'LineWidth', 2
                });
            plot(conds_rel, p_ch(:,i_dif), plot_args1{:});
            hold on;
        end
        hold off;
    end
    
    function plot_rt_by_irr_dif(~, rt, cond_rel, cond_irr, varargin)
        S = varargin2S(varargin, {
            'plot_args', {}
            'y', 'mean' % 'mean'|'var'
            });
        
        %%
        [difs_irr,~,dif_irr] = unique(abs(cond_irr));
        n_dif = length(difs_irr);
        colors = row2cell(hsv2rev(n_dif));
        
        [conds_rel,~,i_cond_rel] = unique(cond_rel);
        n_conds_rel = length(conds_rel);
        
        switch S.y
            case 'mean'
                f = @nanmean;
            case 'var'
                f = @nanvar;
        end
        rt_plot = accumarray([i_cond_rel, dif_irr], rt, ...
            [n_conds_rel, n_dif], f, 0);
        
        for i_dif = 1:n_dif
            plot_args1 = varargin2plot(S.plot_args, {
                '-', ...
                'Color', colors{i_dif}, ...
                'LineWidth', 2
                });
            plot(conds_rel, rt_plot(:,i_dif), plot_args1{:});
            hold on;
        end
        hold off;
    end
    
end
end