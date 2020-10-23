classdef PlotRtDistribAll1D < DtbPlot.PlotPdf1D
    % DtbPlot.PlotRtDistribAll1D
    %
    % 2017 YK wrote the initial version.

%% Properties - Options
properties
    smoothTime = 0.025; % in seconds. Ignored if yfun = 'cumsum'.
    yfun = 'raw'; % 'raw' or 'cumsum'
    
    % y_scale_kind
    % : 'abs': factor * raw_value
    % : 'rel': factor * raw_value / max(abs(raw_value))
    % : 'rel_global': factor * raw_value ...
    %                 / max(abs(raw_value_across_all_conditions))
    %
    % Note
    % : Scale only applies during plotting. Not to get_y.
    %   Use get_y_plot to get y scaled for plot.
    y_scale_kind = 'rel'; 
    y_scale_factor = 1;
end
%% Main
methods
    function Pl = PlotRtDistribAll1D(varargin)
        % Pl = PlotPdf1D(cPdf, plArgs, plotArgs)
        %
        % cPdf(t, cond, ch) : probability mass
        % plArgs: properties of Pl
        % plotArgs: arguments of plot()
        
        Pl = Pl@DtbPlot.PlotPdf1D(varargin{:});
    end
    function [h, ax] = plot(Pl, varargin)
        % h = plot(Pl, varargin)

        % Interpret input
        plotArgs = bml.plot.varargin2plot(varargin, Pl.plotArgs);

        %%
        ax = subplotRCs(Pl.n_cond, 1);
        
        %%
        n_cond = Pl.n_cond;
        n_ch = size(Pl.pdf, 3);
        t = Pl.t;
        p_all = Pl.pdf;
        if strcmp(Pl.yfun, 'cumsum')
            p_all = cumsum(p_all);
%             p_all = p_all ./ nanmax(p_all);
        elseif Pl.smoothTime > 0
            p_all = smooth_gauss(p_all, Pl.smoothTime / Pl.dt);
        end

        h = ghandles(n_cond, n_ch);
        for i_cond = 1:n_cond
            row = n_cond + 1 - i_cond;
            ax1 = ax(row, 1);
            
            for ch = 1:n_ch
                p1 = p_all(:, i_cond, ch);
                p1_all_ch = p_all(:, i_cond, :);
                
                if n_ch == 2
                    sgn = sign(ch - 1.5);
                else
                    sgn = 1;
                end
                
                p1 = p1 .* sgn;
                
                switch Pl.yfun
                    case 'raw'
                        switch Pl.y_scale_kind
                            case 'abs'
                                p1 = p1 .* Pl.y_scale_factor;
                            case 'rel'
                                p1 = nan0(p1 ./ max(abs(p1_all_ch(:)))) ...
                                    .* Pl.y_scale_factor;
                            case 'rel_global'
                                p1 = nan0(p1 ./ max(abs(p_all(:)))) ...
                                    .* Pl.y_scale_factor;
                        end
                    case 'cumsum'
                        p1 = nan0(p1 ./ sum(p1_all_ch(:))) ... % max(abs(p1_all_ch(:)))) ...
                            .* Pl.y_scale_factor;                        
                end
                
                h(i_cond, ch) = plot(ax1, t, p1, plotArgs{:});
                hold(ax1, 'on');
            end
            hold(ax1, 'off');
            crossLine(ax1, 'h', 0);
            bml.plot.beautify(ax1);
            
            if row < n_cond
                set(ax1, 'XTickLabel', ' ', 'YTickLabel', ' ');
            else
                set(ax1, 'YTickLabel', ' ');
                xlabel(ax1, 'Time (s)');
            end
            bml.plot.beautify_tick(ax1, 'x');
            
            y_label = sprintf('%1.4g', Pl.conds(i_cond));
            if Pl.conds(i_cond) == 0
                ylabel(ax1, {'Condition', ' ', y_label});
            else
                ylabel(ax1, y_label);
            end        
        end
    end
end
%% FitWorkspace Interface
methods
    function [hd, hp, Pl_d, Pl_p, ax] = plot_W_pred_data(Pl0, W, varargin)        
        S0 = varargin2S(varargin, {
            'pred_name', 'RT_pred'
            'data_name', 'RT_data'
            });
        S = rmfield(S0, {'pred_name', 'data_name'});
        
        %% Pred
        Pl = feval(class(Pl0));
        Pl.import_W(W, 'src', S0.pred_name);
        C = varargin2plot(S2C(S), {
            'LineStyle', '-'
            'Color', 'r'
            });
        [h, ax] = Pl.plot(C{:});
        Pl_p = Pl;
        hp = h;
        
        %%
        for ii = 1:numel(ax)
            hold(ax(ii), 'on');
        end
        
        %% Data
        Pl = feval(class(Pl0));
        Pl.import_W(W, 'src', S0.data_name);
        C = varargin2plot(S2C(S), {
            'LineStyle', '-'
            'Color', 'k'
            });
        h = Pl.plot(C{:});
        Pl_d = Pl;
        hd = h;
        
        %%
        legend([hp(end,1), hd(end,1)], ...
            {'Prediction', 'Data'}, ...
            'Location', 'northeast');
    end
    function import_W(Pl, W, varargin)
        S = varargin2S(varargin, {
            'src', 'RT_pred' % 'RT_pred', 'Td_pred', 'RT_data'
            });
        
        prop_name = [S.src, '_pdf'];
        if isprop(W.Data, prop_name)
            p = W.Data.(prop_name);
        else
            p = W.Data.(S.src);
        end
        Pl.set_pdf(p);
        Pl.conds = W.Data.conds{1};
        Pl.dt = W.Data.dt;
        
        if ~isempty(strfind(S.src, 'pred'))
            Pl.smoothTime = 0;
        end
    end
end
%% Demo
methods
    function demo(Pl0)
        %%
        file = '/Users/yulkang/Dropbox/CodeNData_2D/ExtRepos/ShadlenLab/Decision2D/Data_2D/Fit.D1.BoundedEn.Main/sbj=DX+prd=RT+tsk=V+dtk=1+dmr=2+trm=201+eor=t+bnd=A+ssq=C+tnd=i+ntnd=2+msf=0+fsqs=0+fbst=1.mat';
        L = load(file);
        L.Fl.res2W;
        W = L.Fl.W;
        
        %%
        Pl = feval(class(Pl0));
        [hd, hp, Pl_d, Pl_p, ax] = Pl.plot_W_pred_data(W);
        
        %%
        
        
    end
end
end