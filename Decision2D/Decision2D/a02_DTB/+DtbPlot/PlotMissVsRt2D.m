classdef PlotMissVsRt2D < DtbPlot.PlotPdf2D
properties
    PlCh
    PlRt
end
    
methods
    function Pl = PlotMissVsRt2D(pdf, varargin)
        Pl = Pl@DtbPlot.PlotPdf2D;
        Pl.PlCh = DtbPlot.PlotCh2D;
        Pl.PlRt = DtbPlot.PlotRt2D;
        
        if nargin >= 1
            Pl.set_pdf(pdf);
        end
        if nargin >= 2
            Pl.init(varargin{:});
        end
    end
    function init(Pl, varargin)
        init(Pl.PlCh, varargin{:});
        init(Pl.PlRt, varargin{:});
    end
    function set_pdf(Pl, pdf)
        Pl.pdf = pdf;
        Pl.PlCh.set_pdf(pdf);
        Pl.PlRt.set_pdf(pdf);
    end
    function set_dimOnX(Pl, d)
        Pl.dimOnX = d;
        Pl.PlCh.set_dimOnX(d);
        Pl.PlRt.set_dimOnX(d);
    end
    function y = get_y(Pl)
        ch = Pl.PlCh.y;
        y  = ch(end,:);
    end
    function x = get_x(Pl)
        rt = Pl.PlRt.y;
        x  = rt(1,:) - rt(end,:);
    end
    function varargout = get_corr(Pl)
        [varargout{1:nargout}] = corr(Pl.x(:), Pl.y(:));
    end
end
methods (Static)
    function Pl = test
        E = Pred.PredInhMcmc;
        E.init;
        E.pred;
        
        %%
        Pl = DtbPlot.PlotMissVsRt2D(E.Td);
        Pl.PlRt.dt = E.dt;
    end
end
end