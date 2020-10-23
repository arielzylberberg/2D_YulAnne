classdef PlotRtByCond < DeepCopyable
    % Fit.D2.Common.Plot.PlotRtByCond
    % 
    % 2016 YK wrote the initial version.
methods
    function [h, res, S] = plot(~, p, cond1, cond2, varargin)
        % p(t, cond1, cond2, ch1, ch2) = P(resp, t, ch1, ch2 | cond1, cond2)
        S = varargin2S(varargin, {
            't', []
            'plot_kind', 'mean'
            'error_kind', 'std'
            'legends', {'LD', 'RD', 'LU', 'RU'}
            });
        assert(isnumeric(p));
        p = squeeze(p(:, cond1, cond2, :, :));
        nt = size(p,1);
        
        p = reshape(p, nt, []);
        
        if isempty(S.t)
            t = (1:nt)';
        else
            t = S.t(:);
        end
        
        switch S.plot_kind
            case 'mean'
                switch S.error_kind
                    case 'sem'
                        [e, m] = sem_distrib(p, t);
                    case 'std'
                        [e, m] = std_distrib(p, t);
                    otherwise
                        error('Unknown error_kind: %s\n', S.error_kind);
                end
                n_cond = length(m);
                h = errorbar(1:n_cond, m, e);
                set(gca, ...
                    'XTick', 1:n_cond, ...
                    'XTickLabel', S.legends);
                
                res = packStruct(t, p, n_cond, m, e);
                
            case 'distrib'
                h = plot(t(:), p);
                legend(h, {'LD', 'RD', 'LU', 'RU'});
                
                res = packStruct(t, p);
        end
    end
    function varargout = plot_W(Plt, W, cond1, cond2, varargin)
        assert(isa(W, 'FitWorkspace'));
        S = varargin2S(varargin, {
            'pdf_kind', 'RT_pred_pdf'
            't', W.get_t
            });
        RT_pdf = W.Data.(['get_' S.pdf_kind]);
        
        C = S2C(S);
        [varargout{1:nargout}] = Plt.plot(RT_pdf, cond1, cond2, C{:});
    end
    function [h, res, S] = plot_W_compare_pred_data(Plt, W, cond1, cond2, varargin)
        for pdf_kind = {'RT_pred_pdf', 'RT_data_pdf'}
            C = varargin2C({
                'pdf_kind', pdf_kind{1}
                }, varargin);
            [h.(pdf_kind{1}), res.(pdf_kind{1}), S.(pdf_kind{1})] = ...
                Plt.plot_W(W, cond1, cond2, C{:});
            hold on;
        end
        hold off;
    end
end
end