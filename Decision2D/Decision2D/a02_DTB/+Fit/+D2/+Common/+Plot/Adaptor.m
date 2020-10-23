classdef Adaptor < Fit.Common.Plot.Adaptor
    % Fit.D2.Common.Plot.Adaptor
    %
    % 2016 YK wrote the initial version.
properties
    dimOnX = 1;
    pdf_kind = 'RT_data_pdf';
    x_bias = [];
end
properties (Transient)
    Data = [];
    p = [];
end
properties (Dependent)
    dimSep
end
methods
    function init(Plt, W, varargin)
        if exist('W', 'var') && ~isempty(W)
            Plt.W = Plt.any2W(W);
        end
        bml.oop.varargin2props(Plt, varargin, true);

        Plt.import_x_bias;
    end
    function import_x_bias(Plt)
        try
            Plt.x_bias = ...
                Plt.W.Dtb.(sprintf('Drift%d', Plt.dimOnX)).th.bias;
        catch
            Plt.x_bias = ...
                Plt.W.Dtb.(sprintf('Dtb%d', Plt.dimOnX)).Drift.th.bias;
        end
    end
    function [p, Dat] = any2p(Plt, inp, pdf_kind)
        % Dat: Fl or W or Dat or pdf
        % pdf_kind: (RT_pred_pdf) | RT_data_pdf | Td_pred_pdf
        % output: always pdf
        
        if isa(inp, 'FitFlow')
            W = inp.W;
            Dat = inp.W.Data;
        elseif isa(inp, 'FitWorkspace')
            W = inp;
            Dat = inp.Data;
        elseif isa(inp, 'FitData')
            Dat = inp;
        elseif isa(inp, 'dataset');
            Dat = Fit.D2.Common.DataChRtPdf;
            Dat.set_ds0(inp);
        end
        
        if isnumeric(inp)
            p = inp;
        else
            assert(isa(Dat, 'FitData'));
            if ~exist('pdf_kind', 'var')
                pdf_kind = Plt.pdf_kind;
            end

            p = Dat.(['get_' pdf_kind]);
            
            if isempty(p)
                if Plt.is_pred(pdf_kind) && exist('W', 'var')
                    W.pred;
                    p = W.Data.(['get_' pdf_kind]);
                else
                    Dat.refresh_RT_data_pdf;
                    p = Dat.(['get_' pdf_kind]);
                end
            end
        end        
    end
%     function tf = is_pred(pdf_kind)
    function import_data(Plt, inp, varargin)
        opt = varargin2S(varargin, {
            'pdf_kind', 'RT_data_pdf'
            });
        [Plt.p, Plt.Data] = Plt.any2p(inp, opt.pdf_kind);
    end
end
methods
    function v = get.dimSep(Plt)
        v = Plt.get_dimSep;
    end
    function v = get_dimSep(Plt)
        N_DIM = 2;
        v = N_DIM + 1 - Plt.dimOnX;
    end
end
end