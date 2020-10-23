classdef PlotMeta < DtbPlot.PlotPdf2D
% Given pdfs (cell array) and spec (dataset), produce meta-plot across pdfs
    
properties
    pdfs = {};
    spec = dataset;
end
properties (Dependent)
    nPdfs
end
methods
    function Pl = PlotMeta(pdfs, varargin)
        Pl = Pl@DtbPlot.PlotPdf2D;
        if nargin >= 1
            Pl.set_pdfs(pdfs);
        end
    end
    function set_pdfs(Pl, v)
        if isa(v, 'Pred.PredMeta')
            v = v.pdfs;
        end
        assert(iscell(v));
        Pl.pdfs = v;
    end
    function v = get.nPdfs(Pl)
        v = length(Pl.pdfs);
    end
    function set_spec(Pl, v)
        assert(isa(v, 'dataset') || istable(v));
        assert(length(v) == Pl.nPdfs);
        Pl.spec = v;
    end
end
end
    