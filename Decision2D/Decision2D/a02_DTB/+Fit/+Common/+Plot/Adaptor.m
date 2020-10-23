classdef Adaptor < DeepCopyable
    % Fit.Common.Plot.Adaptor
    %
    % 2016 YK wrote the initial version.
methods
    function f = Fl2plotfun(~, Fl, funs)
        if ~iscell(funs)
            funs = {funs};
        end

        f = @(x,v,s) plotfun(funs);
        
        function stop = plotfun(funs)
            stop = 0;
            n = numel(funs);
            for ii = 1:n
                funs{ii}(Fl);
                hold on;
            end
            hold off;
        end
    end
    function W = any2W(~, W)
        % W: FitFlow or FitWorkspace
        if isa(W, 'FitFlow')
            W = W.W;
        else
            assert(isa(W, 'FitWorkspace'));
        end
        
    end
    function p = any2p(~, inp, pdf_kind)
        % Dat: Fl or W or Dat or pdf
        % pdf_kind: (RT_pred_pdf) | RT_data_pdf | Td_pred_pdf
        % output: always pdf
        
        Adaptor = Fit.Common.Plot.Adaptor;
        
        if isa(inp, 'FitFlow')
            W = inp.W;
            Dat = inp.W.Data;
        elseif isa(inp, 'FitWorkspace')
            W = inp;
            Dat = inp.Data;
        elseif isa(inp, 'FitData')
            Dat = inp;
        end
        
        if isnumeric(inp)
            p = inp;
        else
            assert(isa(Dat, 'FitData'));
            if ~exist('pdf_kind', 'var')
                pdf_kind = 'RT_pred_pdf';
            end

            p = Dat.(['get_' pdf_kind]);
            
            if isempty(p)
                if Adaptor.is_pred(pdf_kind)
                    W.pred;
                    p = W.Data.(['get_' pdf_kind]);
                else
                    Dat.refresh_RT_data_pdf;
                    p = Dat.(['get_' pdf_kind]);
                end
            end
        end        
    end
    function tf = is_pred(~, pdf_kind)
        tf = ~isempty(strfind(pdf_kind, 'pred'));
    end
end
end