classdef PlotPdf1D < DtbPlot.PlotPdf2D
    % DtbPlot.PlotPdf1D
    %
    % 2015 YK wrote the initial version.
        
methods
%% User interface
function Pl = PlotPdf1D(cPdf, plArgs, plotArgs)
    % Pl = PlotPdf1D(cPdf, plArgs, plotArgs)
    %
    % cPdf(t, cond1, ch1) : probability mass
    % plArgs: properties of Pl
    % plotArgs: arguments of plot()
    if nargin < 1, cPdf = []; end
    if nargin < 2, plArgs = {}; end
    if nargin < 3, plotArgs = {}; end

    % 1D-specific defaults
    Pl.biasDim = 0;
    Pl.condsDim_ = [];
    
    varargin2fields(Pl, plArgs);
    if ~isempty(cPdf)
        Pl.set_pdf(cPdf);
    end
    varargin2fields(Pl, plArgs); % because set_pdf resets conds
    
    Pl.plotArgs = plotArgs;
    if Pl.plotNow || (nargout == 0 && ~isempty(cPdf))
        Pl.plot(plotArgs{:});
    end
end
%% Get/Set - x, y
function v = get_conds(Pl)
    v = Pl.condsDim_;
    if isempty(v)
        n_cond = Pl.n_cond;
        if mod(n_cond, 2) == 0
            v = [-(n_cond/2):-1, 1:(n_cond/2)];
        else
            v = -floor(n_cond/2):floor(n_cond/2);
        end
    end
end
function v = get_n_cond(Pl)
    if isempty(Pl.n_cond_)
        v = size(Pl.pdf_);
        if prod(v) == 0
            Pl.n_cond_ = 0;
        else
            Pl.n_cond_ = v(2);
        end
    end
    v = Pl.n_cond_;
end
function set_conds(Pl, v)
    try
        assert(isnumeric(v));
        assert(isvector(v));
    catch err
        warning('PlotPdf1D.conds must be a numeric vector!');
        rethrow(err);
    end
    Pl.condsDim_ = v;
end
function v = get_condsDim(Pl)
    v = Pl.get_conds;
end
function set_condsDim(Pl, v)
    Pl.set_conds(v);
end
function v = get_condsAxis(Pl)
    v = Pl.get_conds;
end
function set_condsAxis(Pl, v)
    Pl.set_conds(v);
end
%% Get/Set - bias
function bias = get_x_bias(Pl)
    bias = Pl.biasDim;
end
function set_x_bias(Pl, bias)
    Pl.biasDim = bias;
end
function v = get_biasDim(Pl)
    v = Pl.biasDim_;
end
%% Set properties
function set_pdf(Pl, pdf)
    assert(isnumeric(pdf));
    assert(any(ndims(pdf) == 3));
    
    Pl.pdf_ = pdf;
    
%     assert(any(ndims(pdf) == [3 5]));
%     if ndims(pdf) == 3
%         % Expand dimensions
% %         if Pl.dimOnX == 1
%             pdf = permute(pdf, [1 2 4 3]);
% %         else
% % %             pdf = permute(pdf, [1 4 2 3]);
% %         end
%         pdf = cat(5, zeros(size(pdf)), pdf);
%     end
%     
%     Pl.set_pdf@DtbPlot.PlotPdf2D(pdf);
end
end
end % classdef