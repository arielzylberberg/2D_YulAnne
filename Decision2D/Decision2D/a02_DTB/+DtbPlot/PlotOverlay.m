classdef PlotOverlay < DtbPlot.PlotMeta
properties
    CompoPl = [];
end
methods
    function Pl = PlotOverlay(varargin)
        Pl = Pl@DtbPlot.PlotMeta(varargin{:});
    end
    function set_CompoPl(Pl, CompoPl)
        assert(isa(Pl.CompoPl, 'DtbPlot.PlotPdf2D'));
        Pl.CompoPl = CompoPl;
    end
    function plot(Pl, varargin)
        assert(isa(Pl.CompoPl, 'DtbPlot.PlotPdf2D'));
        colors = colors2mat(Pl.colors, Pl.nPdfs);
        CompoPl = Pl.CompoPl;
        
        % Reuse plot function of CompoPl
        for iPdf = 1:Pl.nPdfs
            CompoPl.set_pdf(Pl.pdfs{iPdf});
            CompoPl.set_colors(colors(iPdf,:));
            CompoPl.plot(varargin{:});
            hold on;
        end
        hold off;
    end
end
end