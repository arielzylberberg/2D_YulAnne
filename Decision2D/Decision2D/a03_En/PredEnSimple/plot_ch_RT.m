function ax = plot_ch_RT(sTr, varargin)
    S = varargin2S(varargin, {
        });

    ch = sTr.ch;
    rt = sTr.RT;
    cond = sTr.cond;
    
    n_row = 2;
    n_col = 2;
    n_dim = 2;
    Pl = Plot2DSimple;
    dim_names = {'Motion', 'Color'};
    
    ax = subplotRCs(n_row, n_col);
    
    for dim_rel = 1:n_dim
        dim_irr = setdiff(1:n_dim, dim_rel);

        axes(ax(1, dim_rel));
%         subplotRC(n_row, n_col, 1, dim_rel);
        Pl.plot_ch_by_irr_dif(ch(:,dim_rel), cond(:,dim_rel), ...
            cond(:,dim_irr));
        
        title(dim_names{dim_rel});
        if dim_rel == 1
            ylabel('P(ch=2)');
        end
        bml.plot.beautify;
    
        axes(ax(2, dim_rel));
%         subplotRC(n_row, n_col, 2, dim_rel);
        incl = ~isnan(ch(:, dim_rel));
        Pl.plot_rt_by_irr_dif(rt(incl), cond(incl,dim_rel), ...
            cond(incl,dim_irr));
        
        if dim_rel == 1
            ylabel('RT (s)');
        end
        bml.plot.beautify;
    end    
end