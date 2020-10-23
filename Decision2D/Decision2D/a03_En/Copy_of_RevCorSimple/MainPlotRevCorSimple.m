classdef MainPlotRevCorSimple < matlab.mixin.Copyable
methods
    function S1 = plot_by_dim_align(Plot, tbl, S0, varargin)
        S = varargin2S(varargin, {
            'ax', []
            });
        
        n_dim = numel(S0.dim_rel);
        n_align = numel(S0.align);
        
        dif_rel = S0.dif_rel{1};
        dif_irrs = S0.dif_irr;
        n_dif_irrs = numel(dif_irrs);
        aligns = cell2mat(S0.align);
        subj = S0.subj{1};
        
        colors = row2cell(hsv2rev(n_dif_irrs));
        
        n_row = n_dim;
        n_col = n_dim * n_align;
        if isempty(S.ax)
            ax = subplotRCs(n_row, n_col);
        else
            ax = S.ax;
        end
        
        for dim_rel = 1:n_dim
            for i_dif_irr = 1:n_dif_irrs
                for n_dim_task = 1:n_dim
                    for i_align = 1:n_align
                        align = aligns(i_align);

                        col = (n_dim_task - 1) * n_dim + i_align;
                        row = dim_rel;
%                         col = (dim_rel - 1) * n_align + i_align;
%                         row = n_dim_task;
                        axes(ax(row, col));
                        
                        S1 = varargin2S({
                            'align', align
                            'subj', subj
                            'n_dim_task', n_dim_task
                            'dim_rel', dim_rel
                            'dif_rel', dif_rel
                            'dif_irr', dif_irrs{i_dif_irr}
                            }, S0);
                        Plot.plot_align(tbl, S1, ...
                            'color', colors{i_dif_irr});
                                                
                        
                        if row < n_row || col > 2
                            xlabel('');
                            set(gca, 'XTickLabel', '');
                        end
                        if col > 1
                            ylabel('');
                            set(gca, 'YTickLabel', '');
                        end
                        if i_align == 1 && row == 1
                            title(sprintf('%dD', n_dim_task));
                        end
                        
%                         xlim([0, 2]);
                    end
                end
            end
        end

        for i_col = 1:n_col
            sameAxes(ax(:, i_col), [], [], 'x');
        end
        for i_row = 1:n_row
            sameAxes(ax(i_row, :), [], [], 'y');
        end        
    end
    function tbl1 = plot_align(Plot, tbl, S0, varargin)
        S = varargin2S(varargin, {
            'color', [0, 0, 0]
            'time_shift', 0
            });
        incl = (tbl.align == S0.align) ...
             & strcmp(S0.subj, tbl.subj) ... & cellfun(@(a) isequal(a, S0.subj), tbl.subj) ...
             & (tbl.n_dim_task == S0.n_dim_task) ...
             & (tbl.dim_rel == S0.dim_rel) ...
             & cellfun(@(v) ...
                 isequal(S0.dif_irr, v), ...
                 row2cell2(tbl.dif_irr)) ...
             & cellfun(@(v) ...
                 isequal(S0.dif_rel, v), ...
                 row2cell2(tbl.dif_rel));

        tbl1 = table2struct(tbl(incl, :));

        %% Plot
        if any(incl)
            C = varargin2C({'time_sign', -S0.align}, S);
            Plot.plot(tbl1, C{:}); 
        else
            ylim([0, eps]);
        end
        hold on;

        if S0.align == 1
%             set(gca, 'XDir', 'reverse');
            xlabel('time from RT (s)');
        else
            xlabel('time from onset (s)');
        end
        if S0.dim_rel == 1
            ylabel('Motion');
        elseif S0.dim_rel == 2
            ylabel('Color');
        end
        bml.plot.beautify;
        crossLine('h', 0, 'k:');
    end
    function plot(Plot, tbl1, varargin)
        % tbl1: struct with the following fields 
        % (as from MainRevCorSimple.main)
        % .slope(fr)
        % .se_slope(fr)
        % .wt(fr) # fraction of trials * frames included
        % .t(fr)
        %
        % OPTIONS:
        % 'color', [0, 0, 0]
        S = varargin2S(varargin, {
            'color', [0, 0, 0]
            'time_shift', 0
            'time_sign', 1
            });
        %%
        slope = tbl1.slope;
        se_slope = tbl1.se_slope;
%         bias = tbl1.bias;
%         se_bias = tbl1.se_bias;
        wt = tbl1.nanmean_wt;
        t = tbl1.t(:);

        t_incl = wt > 0.75;
        slope1 = slope(t_incl);
        se_slope1 = se_slope(t_incl);
        errorbarShade(S.time_sign * t(t_incl) + S.time_shift, ...
            slope1, se_slope1, ...
            S.color); 
    end
end
end