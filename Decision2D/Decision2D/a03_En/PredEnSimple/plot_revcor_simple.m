function ax = plot_revcor_simple(tbl, S0, varargin)
    S = varargin2S(varargin, {
        'subj', S0.subj{1}
        'n_dim_task', 2
        });

    Plot = MainPlotRevCorSimple;
    ax = subplotRCs(2,2);

    aligns = [-1, 1];
    n_align = numel(aligns);
    n_dim_task = S.n_dim_task;
    colors = hsv2rev(2);

    y_lim = repmat([inf, -inf], [2, 1]);
    
    for dim_rel = 1:2
        for i_align = 1:n_align
            align = aligns(i_align);

            for i_dif_irr = 1:numel(S0.dif_irr)
                dif_irr = S0.dif_irr{i_dif_irr};
                S1 = varargin2S({
                    'align', align
                    'subj', S.subj
                    'n_dim_task', n_dim_task
                    'dim_rel', dim_rel
                    'dif_rel', S0.dif_rel{1}
                    'dif_irr', dif_irr
                    }, S0);

                axes(ax(dim_rel, i_align));
                Plot.plot_align(tbl, S1, 'color', colors(i_dif_irr, :));
                hold on;
            end
            
            y_lim1 = bml.plot.beautify_lim;
            y_lim(dim_rel, 1) = min([y_lim(dim_rel, 1), y_lim1(1)]);
            y_lim(dim_rel, 2) = max([y_lim(dim_rel, 2), y_lim1(2)]);
            
%             ylim([0, 0.06]);
%             xlim([0, 1]);
            if dim_rel == 1
                set(gca, 'XTickLabel', '');
                xlabel('');
            end
            if i_align == 2
                set(gca, 'YTickLabel', '');
                ylabel('');
            end
        end
    end
    for dim_rel = 1:2
        for i_align = 1:n_align
            axes(ax(dim_rel, i_align));
            ylim(y_lim(dim_rel, :));
        end
    end
    

    bml.plot.position_subplots(ax);
end