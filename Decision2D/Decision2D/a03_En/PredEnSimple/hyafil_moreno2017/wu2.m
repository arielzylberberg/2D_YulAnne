function [M, L, plothandle] = wu2(varargin)
% WU is a very handy and customizable function for plotting curves and bars with error bars.
%
%* BASIC SYNTAX:
% wu(Y) or wu(Y, 'mean') if Y is a m-by-n numerical matrix will plot a n-point curve with 1 to n integers along X-axis, and the
% mean of each row in Y along Y-axis, together with error bars centered on the mean whose length
% is the standard variation for each row in Y. Nan values are ignored.
% If Y is a 3-dimension matrix, a curve with errors bars will be plotted
% for each submatrix Y(:,:,i).
% If Y is a 4-dimension or 5-dimension matrix, figure is divided in subplot
% corresponding to the different Y(:,:,:,k,l) submatrices.
% If Y is a cell array whose elements is composed of numerical vectors, mean and
% standard errors in computed within each array. A single curve with error
% bars will be drawn for vectorial cell array, multiple for 2-dimensional
% cell arrays, and multiple subplots for 3- and 4-dimension cell arrays.
%
% wu(Y,'median') uses medians of each row instead of means.
%
% wu(Y,errortype) specifies how error values are computed. Possible values
% are:
% - 'std' :standard deviation (default option for 'mean')
% - 'ste': standard error of the mean
% - 'correctedste': standard error of the mean removing mean value over
% all data in the same line (i.e. removing variance due to random factor),
% provides a visual hint of significance of t-test
% - 'quartile': plots 1st and 3rd quartile
% - 'noerror' to simply plot mean values without errors.
%
% wu(M,E) allows to directly specify the values for the means and errors. M
% and E must be numerical matrices (up to 4 dimensions) of the same size.
% Use wu(M,[]) to only plot mean values (with no error bars).
%
% wu(X,Y) or wu(X,M,E) allows to specify in vector X the values along the
% X-axis.
%
% wu([],M,L,U) or wu(X,M,L,U) allows to specify in L and U the length of bars below and
% above its centers, for asymmetrical error bars.
%
% wu(..., plottype) defines plot type between the following:
% - 'curve' draws a simple curve joining values for mean (default option)
%  - 'bar' draws a bar plot with one bar for each mean value
%  - 'xy' compares values in first column of M in X-axis against value in
%  second column in M axis. M must have 2 rows ( but can be more than
%  2-dimensional), or  Y must have two columns.
%  - 'imagesc' displays matrix M as an image (where M is at least 2 dimensional, or equivalently Y is 3-dimension or more). Error values are not plotted.
%
% wu(...., 'curve','errorstyle',errorstyle) defines how error are plotted
% when mean values are plotted as a curve. Possible values are:
% - 'bar' : error bars centered on mean value (default when less than 6
% data points per curve)
% - 'curve' : two curves above and below the main curve, join respectively
% points M+E and points M-E (default when 6 or more data points per curve)
% - 'area' : shades an area defines betwen points M-E and M+E
%
% wu(..., {variablename1 variablename2 ...}) specifies labels for the
% independent factors in matrix Y (or M and E), i.e. labels for dimension
% 1, 2, etc. in matrix M. This is used to label X-axis, add legend and
% title. Use '' for a variablename to avoid naming it.
%
% wu(...., { {v1label1 v1label2...},{v2label1 v2label2...}) specifies labels for
% the different levels within each variable, i.e. for the different rows,
% columns, etc. of matrix M (corresponding to columns of Y, elements Y(:,:,i), Y(:,:,:,i), etc.).
% These labels are used to set ticknames along X-axis, legend, and
% subtitles. Leave Use {} for a variable to avoid labelling its level.
%
% wu(...., 'pvalue',P) where P is a numerical matrix of p-values of size equal to that of M
% allows to draw lines to signal points reaching statistical significance.
% wu(Y, 'pvalue','on') or wu(X,Y, 'pvalue','on') computes p-value by
% applying two-sided t-test on the columns of Y.
% wu(....,'threshold',alpha) adjusts the threshold for statistical
% significance (by default 0.05).
% wu(....,'correction','on') uses Bonferroni correction for multiple
% comparison on threshold, i.e. uses as threshold alpha/numel(M).
%
%* PLOT SPECS:
% wu(...., specname, specvalue) allows to specify plot options. Available
% specnames are:
% -'ylabel': label for dependent variable (used to label Y-axis)
% - 'color' : defines colour for mean values (bars or curves). Value can
% be: a single color (either RGB vector or character such as 'k' or 'b')
% sue the same color for each plot; a cell array of colours (each element
% being RGB or symbol) or a n x 3 matrix to associate a different color
% with each curve/bar series (i.e. each column of M); a string array of
% a built-in colormap (e.g. 'jet', 'hsv', 'gray') to use graded colors
% within the colormap; or 'flat' to use graded colors from the current
% colormap (allows to use user-defined colormap).
% - 'errorstyle' : defines how error values E are represented. Possible
% values are 'bar' (an error bar associated with each value, default option), 'line' (a
% dashed error line plotted above the mean line at M+E, and another below at M-E),
% and 'fill' (a light colour surface covering all area between M-E and M+E).
% - 'axis' : defines on which axis dependent measures (M and E) are
% plotted. Default is 'y'. Use 'x' to swap axes (e.g. for horizontal bars).
% - 'legend': use 'color' to use coloured text for legend (default),
% 'standard' to use matlab built-in legend style, or 'none' to avoid
% plotting legend
%
% !!check that it works
% 'ErrorBarStyle': line style for error bars.
% 'ErrorBarColor': RGB or character controlling color of error bars (including ticks).
% 'ErrorBarWidth': controls the width of the line composing error bars (including ticks).
% 'ErrorBarMarker': controls the marker placed at each end of error bars.
% 'ErrorBarMarkerSize': controls the size of the marker placed at each end of
% error bars.
% 'ErrorBarMarkerFaceColor': control the color of markers at each end of error
% bars
% 'TickLength' : controls the length of the horizontal ticks at each end of the bars.
% Set to 0 to avoid plotting ticks.
% 'TickColor': RGB or character controlling color of ticks only.
% 'TickWidth': controls the width of the line composing ticks.
% 'XTickRotate' : rotates Xtick labels by specified angle (for matlab
% version prior to 2014b, requires function rotateXlabels)
%
% Options for bar style only:
% 'BaseValue': baseline location for bars (scalar, default:0).
% 'Barwidth' : relative width of individual bars (scalar between 0 and 1, default 0.8)
% 'FaceColor': Bar fill color, can take values: 'flat' (default) | 'none' | RGB triplet | color string
% 'LineColor': Defines color for all lines: bar outlines, error bars and
% ticks. Can take values: 'flat' | 'none' | RGB triplet | color string.
% 'LineWidth': Scalar defining width for all lines: bar outlines, error bars and
% ticks.
% 'EdgeColor': Specifically bar outline color, can take values : 'flat' | 'none' | RGB triplet | color string
% 'Edgestyle': Line style for bar outline (default: continuous line '-')
% 'VerticalLabel': if value is set to 'on', places labels as text over each
% bar instead along the axis (only for single bar series)
%
% Options for 'curve'  and 'area' error style only:
% 'errorlinestyle' : line style for curves at M+E and M-E (default: '--'
% for 'curve' error style, 'none' for 'area' error style)
%
% Options for 'xy' plot only:
% 'IdentityLine' : plots identity line (i.e. line x=y) if set to true
% (default:false)
%
% Options for imagesc only:
% 'clim':
%
%* OUTPUT ARGUMENTS:
% [M E h] = wu(...). h is the plot handle for the different graphic
% objects
%
% See also mybar, myerrorbar

%TODO:
% check help for options
% add check for xy ( just 2 variables)
% use wu(X, M,U,D) for asymetric error bars
% add parameters for myerrorbar and mybar
% error bars for binary data : http://stats.stackexchange.com/questions/11541/how-to-calculate-se-for-a-binary-measure-given-sample-size-n-and-known-populati
% change names for nbp and nbv
% option 'single' to have just one bar series/curve per subplot.
%clean-up and graphic handles
%'xy' : allow 3D matrix (2xAxB) on single subplot
% check imagesc
% look back at p values. use 'barwebpairs' for displaying significance
% add line specs for line.
% xtickrotation (see commented lines)
% add 'offset' to have non-overlapping error bars

%
%
%OPTIONS :
%'xtick'     : 'normal' to place X values according to values of first
%parameter levels (if numerical) ; 'index' to use regular spacing of levels
%along X axis
%'axis' : on which axis dependent measure is plotted along: 'Y' (e.g. plot vertical bars, default value) or 'X' (underconstruction)
%'pvalue'    : p-values, same size as Ymean
%'permute'   : a vector to permute the dimensions for plotting (deprecated)



%default values
X = [];
L = [];
U = [];
cor =  defaultcolor;
clim = 'auto';
linestyle = '';
marker = '.';
markersize = 6;
linewidth = 2;
%ticklength = [];
plottype = 'curve';
edgecolor = '';
linecolor = '';
errorstyle = '';   % '', {'bar', 'bars', 'errorbar', 'errorbars'}, {'fill', 'area', 'marge'},  {'line', 'lines', 'errorline', 'errorlines'}
errorlinestyle = ''; % default style for curve errors
pval = [];
threshold = .05;%permut = [];
correction = 'off';
titl = '';
y_label = '';
name ='';
doclf = 'noclf';
xtick = 'normal'; %'normal' or 'index' (does not use level values even if real numbers) / i think its the same as 'factordirection' parameter in boxplot
maxis = 'y'; % axis for dependent variable
xtickangle = 0; % angle of rotation for xtick labels
VerticalLabel = 0; % places labels for first variable as vertical text above bars
legend_style = 'color';
IdentityLine = 0; % identity line for 'xy' plot

%% determine syntax
numargs = cellfun(@isnumeric, varargin);  %which arguments are numeric
numargs(5) = false;                 %in case there is less than 3 args
syntax = find(numargs==0,1)-1; % number of consecutive numeric arguments
%syntax = sum(numargs(1:3));

if syntax<=2 && nargin > syntax && (syntax==0 || isvector(varargin{syntax})) && iscell(varargin{syntax+1}) && isnumeric(varargin{syntax+1}{1}),
    syntax = syntax + 1;   % for cell arrays of values
    rawinput = 1;
else
    rawinput = 0;
end

do_median = any(cellfun(@(x) isequal(x,'median'), varargin(syntax+1:end))); % whether to use median
if do_median
    avg_fun = @nanmedian;
    errorbars = 'quartile'; % default option
else
    avg_fun = @nanmean;
    errorbars = 'ste';
end

switch syntax
    case 1  % just raw data : wu(Y, ...)
        Yraw = varargin{1};
        if ~rawinput    %N-D array : mean over first dimension
            M = avg_fun(Yraw,1);
            M = shiftdim(M,1);
        else % wu(Ycell, ...)
            M = cellfun(@(x) avg_fun(x,1), Yraw); %N-D cell array : mean within cells
        end
    case 2
        if rawinput % wu(X, Ycell,...)
            X = varargin{1};
            Yraw = varargin{2};
            M = cellfun(avg_fun, Yraw); %N-D cell array : mean within cells
            
        elseif isequal(size(varargin{1}), size(varargin{2}))  % wu(Ymean, Yerror, ...)
            M = varargin{1};
            L = varargin{2};
            U = L;
        elseif isempty(varargin{2})   % wu(Ymean, [], ...) ; mean, no error bars
            M = varargin{1};
            L = NaN(size(M));
            U = L; %symmetric bars
        else % wu(X, Y)
            X = varargin{1};
            Yraw = varargin{2};
            M = avg_fun(Yraw,1);
            M = shiftdim(M, 1); % pass on dimension 2 to dimension 1, dim 3 to dm 2, etc.            
        end
    case 3 % wu(X, M, E)
        X = varargin{1};
        M = varargin{2};
        if ~isempty(varargin{3})
            L = varargin{3};
        else  % no error bar
            L = NaN(size(M));
        end
        if isrow(M) && length(X)>1 % if M is provided as row instead of column, correct
            M = M';
            L = L';
        end
        U = L; %symmetric bars
    case 4, % case(X, M, L, U)
         X = varargin{1};
        M = varargin{2};
        if ~isempty(varargin{3})
            L = varargin{3};
        else  % no error bar
            L = NaN(size(M));
        end
        if ~isempty(varargin{4})
            U = varargin{4};
        else  % no error bar
            U = NaN(size(M));
        end
        if isrow(M) && (isempty(X) || length(X)>1) % if M is provided as row instead of column, correct
            M = M';
            L = L';
            U = U';
        end
end

if isempty(M)
    plothandle = [];
    return;
end

% check that data is real
if ~all([isreal(X) isreal(M) isreal(L) isreal(U)])
    warning('MATLAB:plot:IgnoreImaginaryXYPart','Imaginary parts of complex X and/or Y arguments ignored');
end
imaginary_warning = warning('off','MATLAB:plot:IgnoreImaginaryXYPart'); % turn it off for all called functions


%size and dimensionality
siz = size(M);
dimm = length(siz);
nbp = size(M,1); % number of parameters in M
nbv = size(M,2); % number of variables in M



%%

%default names for variables and levels
varnames = repmat({''}, 1, dimm);
levels = cell(1, dimm);
for d = 1:dimm
    if d==1 && ~isempty(X)
        levels{d} = num2strcell(X);
    elseif siz(d)>1
        levels{d} = num2strcell(1:siz(d));
    else % if only one level, no need to add a label
        levels{d} = {};
    end
end


%% %%%% OPTION INPUTS %%%%%
errorbarpars = {};
v = syntax+1; % where options start
while v<=length(varargin)
    varg = varargin{v};
    switch class(varg)
        case 'char'
            switch(lower(varg))
                case {'mean','median'}
                 % already processed before
                case {'color','facecolor'}
                    v = v +1;
                    cor = varargin{v};
                case 'style'
                    v = v + 1;
                    marker = varargin{v};
                case 'linewidth'
                    v = v+1;
                    linewidth = varargin{v};
                    %                 case 'ticklength',
                    %                     v = v+1;
                    %                     ticklength = varargin{v};
                case   {'barwidth','errorbarstyle','errorbarwidth','errorbarcolor', 'errorbarmarker','errorbarmarkersize','errorbarmarkerfacecolor',...
                        'tickcolor','ticklength', 'tickwidth','errorbar','basevalue'}
                    errorbarpars(end+1:end+2) = varargin(v:v+1);
                    v = v + 1;
                case 'linecolor'
                    v = v+1;
                    linecolor = varargin{v};
                case 'edgecolor'
                    v = v+1;
                    edgecolor = varargin{v};
                case 'errorbarplottype'
                    v = v+1;
                    errorbars = varargin{v};
                case 'errorlinestyle'
                    v = v+1;
                    errorlinestyle = varargin{v};
                case 'pvalue'
                    v = v+1;
                    pval = varargin{v};
                case 'threshold'
                    v = v+1;
                    threshold = varargin{v};
                case 'correction'
                    v = v+1;
                    correction = varargin{v};
                case 'ylabel'
                    v = v+1;
                    y_label = varargin{v};
                case 'title'
                    v = v+1;
                    titl = varargin{v};
                case 'name'
                    v = v+1;
                    name = varargin{v};
                case 'errorstyle'
                    v = v+1;
                    errorstyle = varargin{v};
                case 'marker'
                    v = v+1;
                    marker = varargin{v};
                case 'markersize'
                    v = v+1;
                    markersize = varargin{v};
                case 'legend'
                    v = v+1;
                    legend_style = varargin{v};
                case 'linestyle'
                    errorbarpars(end+1:end+2) = varargin(v:v+1); % if 'bar' type
                    v = v+1;
                    linestyle = varargin{v}; % if 'curve' type
                    
                    %                 case defcolor,
                    %                     cor = {varg};
                case {'-',':','-.','--','none'}
                    linestyle = varg;
                case {'.','o','x','+','*','s','d','v','^','<','>','p','h'}
                    marker = varg;
                case {'ste', 'std', 'correctedste', 'quartile','noerror'}
                    errorbars = varg;
                case {'curve', 'bar', 'xy', 'imagesc'}  %'noplot'
                    plottype = varg;
                case 'line'
                    plottype = 'curve';
                case {'jet', 'gray', 'pink', 'hsv', 'hot', 'cool', 'copper', 'flag', 'prism'}
                    cor = eval(varg);
                case 'clim'
                    v = v +1;
                    clim = varargin{v};
                case {'clf', 'noclf'}
                    doclf = varg;
                case 'permute'
                    error('not supported anymore');
                    v = v+1;
                    permut = varargin{v};
                case 'xtick'
                    v = v+1;
                    xtick = varargin{v};
                case 'axis'
                    v = v+1;
                    maxis = varargin{v};
                case 'xtickrotate'
                    v = v +1;
                    xtickangle = varargin{v};
                case 'verticallabel'
                    v = v+1;
                    VerticalLabel = strcmpi(varargin{v},'on');
                case 'identityline'
                    v = v +1;
                    IdentityLine = varargin{v};
                otherwise
                    % check whether it is line specification
                    [XL,XC,XM,MSG] = colstyle(varg);
                    if ~isempty(MSG)
                        error('incorrect parameter: %s', varg);
                    end
                    % fill in non empty line specs
                    if ~isempty(XL)
                        linestyle = XL;
                    end
                    if ~isempty(XC)
                        cor = XC;
                    end
                    if ~isempty(XM)
                        marker = XM;
                    end
            end
        case 'cell'
            if ~isempty(varg) && ischar(varg{1})  % labels for variables
                %varnames = varg;
                addvarnames = ~cellfun(@isempty,varg);
                varnames(addvarnames) = varg(addvarnames);
            else  % label for values / levels
                addlevels = ~cellfun(@isempty,varg);
                levels(addlevels) = varg(addlevels);
                %  else
                %      error('unknown input');
            end
        otherwise
            error('incorrect option class: %s', class(varg) );
    end
    v = v +1;
end

% 'xy' option : last dimension must have two values
if strcmp(plottype, 'xy') && nbp ~=2
    error('wu : for ''xy'' option, size of the first dimension of data matrix must be 2');
end

% factor axis (if measure axis is Y, then it is X, and vice versa)
if strcmp(maxis, 'x')
    faxis = 'y';
else
    faxis = 'x';
end



%% process error bar values
if strcmp(plottype, 'imagesc')
    errorbars = 'noerror';
end
if isempty(L)
    if isnumeric(Yraw)
        switch errorbars
            case 'ste'
                L = nanste(Yraw,1);
            case 'std'
                L = nanstd(Yraw,1);
            case 'correctedste',
                Ym = reshape(Yraw, [size(Yraw,1) prod(siz)]); %
                Ym = mean(Ym, 2);  % means across all conditions
                L = nan_ste(bsxfun(@minus,Yraw,Ym)); %remove these means from individual values
            case 'quartile'
                L = quantile(Yraw, .25); % 25th percentile
                L = M - reshape(L, size(M)); % remove mean/median
                 U = quantile(Yraw, .75); % 75th percentile
                U = reshape(U, size(M)) - M;
            case 'noerror',
                L = NaN(siz);
        end
        if ~strcmp(errorbars,'quartile')
            L = shiftdim(L,1);
            U = L;
        end
    else
        switch errorbars
            case 'ste'
                L = cellfun(@ste, Yraw);
                        U = L;
            case 'std'
                L = cellfun(@std, Yraw);
                        U = L;
            case 'correctedste'
                warning('cannot correct for random factor variance if data is not paired, using classical standard error instead');
                L = cellfun(@ste, Yraw);
                        U = L;
            case 'quartile'
                L = M - cellfun(@(x) quantile(x,.25), Yraw);
                               U = cellfun(@(x) quantile(x,.75), Yraw) - M;
            case 'noerror'
                L = NaN(size(M));
                        U = L;
        end
    end
end

%% parse labels and variable names

%!! check with default names given above

%!! to be removed ? (mystats)
removeunderscore = 0;
if removeunderscore
    varnames = strrep(varnames, '_', ' '); %remove '_'
end

%if more dimensions than thought (because of singleton dimensions), append
if length(levels)>dimm
    newdimm = length(levels);
    siz(dimm+1:newdimm) = 1;
    dimm = newdimm;
end


%labels for levels
for i=1:length(levels)
    %check size
    if  ~isempty(levels{i}) && length(levels{i})~=siz(i)
        error('Numbers of labels for variable along dimension %d (%d) does not match number of levels (%d)',i, length(levels{i}),siz(i));
    end
    %     if  size(levels{i})>siz(i),
    %         warning('labels for variable along dim %d is %d, higher than number of levels (%d); extra ones will not be used',i, length(levels{i}),siz(i));
    %     end
    
    %if level array is numeric array, turn into cell array of strings
    if isnumeric(levels{i})
        levels{i} = num2strcell(levels{i});
    end
    
    % replace '_' chars
    if removeunderscore
        levels{i} = strrep(levels{i}, '_', ' ');
    end
end

if VerticalLabel && ~((strcmp(plottype, 'curve')&&nbv~=1) ||  (strcmp(plottype, 'bar')&&any([nbv nbp]~=1)))
    error('vertical labels only for single series bar/curve type');
end

% %% permute if required
% if ~isempty(permut)
%     M = permute(M, permut);
%     E = permute(E, permut);
%     varnames(2:end) = varnames(1+permut);
%     levels = levels(permut);
%     siz = siz(permut);
% end

%% use values for xtick if set 'normal'
if strcmp(xtick, 'normal') && ~isempty(levels{1}) && isempty(X)
    % try to convert all labels into X value
    X = cellfun(@str2double, levels{1}(:)');
    if any(isnan(X)) || isempty(X) % if its fails, simply take first integers
        X = 1:nbp;
    end
elseif isempty(X)
    X = 1:nbp;
end



%% process p-values
if isempty(pval)
    pval = nan(size(M));
    % pval = 1;
    % if dimm == 3, pval = nan(1, siz(3)); end
    % if dimm == 4, pval = ones(siz(3), siz(4)); end
elseif ischar(pval) && strcmp(pval,'on') % compute p-value from t-test applied to Y
    if ~exist('Yraw','var')
        error('to compute p-value, syntax must be with raw array Y');
    end
    if ~rawinput % if numerical array, convert to cell array with one column
        Yraw = num2cell(Yraw,1);
        Yraw = shiftdim(Yraw, 1); % pass on dimension 2 to dimension 1, dim 3 to dm 2, etc.
    end
    [~, pval] = cellfun(@ttest, Yraw); % apply T-test
else
    if~isequal(size(pval),size(M))
        error('pvalue should have the same size as mean values');
    end
end
if strcmp(correction, 'on') % 'Bonferroni correction',
    threshold = threshold / numel(M);
end
ymax = max( max(M(:) +abs(U(:))), max(M(:)) ); % maximum value of mean point or error
ymin = min( min(M(:) -abs(L(:))), min(M(:)) ); % minimum value of mean point or error
if isnan(ymin) || ymin==0
    ymin = 1;
end
y_pval = ymax + (.05 + .02*(0:nbv-1)/(nbv-1))*(ymax-ymin); % vertical positioning of significance lines

%% default linestyle
if isempty(linestyle)
    if strcmp(plottype, 'xy')
       linestyle = 'none';
    else
        linestyle = '-';
    end
end

%% parse error style
if isempty(errorstyle)
    if nbp<=5
        errorstyle = 'bar';
    else
        errorstyle = 'curve';
    end
end

%% parse colors into n-by-3 RGB matrix
cor = color2mat(cor,nbv);
if ~isempty(edgecolor)
    edgecolor = color2mat(edgecolor,nbv);
    errorbarpars(end+1:end+2) = {'EdgeColor',edgecolor};
end
if ~isempty(linecolor)
    linecolor = color2mat(linecolor,nbv);
    errorbarpars(end+1:end+2) = {'LineColor',linecolor};
end
% if ischar(cor),
%     switch lower(cor),
%         case  {'jet', 'gray', 'pink', 'hsv', 'hot', 'cool', 'copper', 'flag', 'prism'},
%             cor = eval(cor);
%             if nbv>1
%                 corvec = 1 + (size(cor,1)-1) * (0:nbv-1)/(nbv-1);
%             else
%                 corvec = 1;
%             end
%             cor = cor(floor(corvec),:);
%         case {'flat','colormap'}, %interpolate colors from colormap
%             cor = colormap;  %current colormap
%             if nbv>1
%                 corvec = 1 + (size(cor,1)-1) * (0:nbv-1)/(nbv-1);
%             else
%                 corvec = 1;
%             end
%             cor = cor(floor(corvec),:);
%         otherwise  %just one colour symbol (e.g. 'k')
%             cor = {cor};
%     end
% end
%
% % if cell array, convert to matrix of RGB values
% if iscell(cor),
%     cor_mat = zeros(length(cor),3);
%     for c=1:length(cor),
%         if isnumeric(cor{c}), % RGB value
%             cor_mat(c,:) = cor{c};
%         elseif ischar(cor{c}), % letter (e.g. 'k', 'b', etc.)
%             cor_mat(c,:) = rem(floor((strfind('kbgcrmyw', cor{c}) - 1) * [0.25 0.5 1]), 2);
%         else
%             error('incorrect value for colour: should be vector of RGB value or single character');
%         end
%     end
%     cor = cor_mat;
% end

% if not enough colours compared to number of variables,
% cycle through them
if size(cor,1) < nbv
    cor =  cor( mod(0:nbv-1,size(cor,1))+1 ,:  );
end

% compute light colours for margin
if strcmp(plottype, 'curve') && any(strcmp(errorstyle, {'fill', 'area', 'marge'})),
    lightcolour = .8+.2*cor; %light colors
end

if isempty(errorlinestyle)
    switch errorstyle
        case  {'fill', 'area', 'marge'} %error areas
            errorlinestyle = 'none'; % default style for curve errors
        case {'curve', 'curves', 'errorcurve', 'errorcurves'} %error lines
            errorlinestyle = '--'; % default style for curve errors
    end
end

%if ~strcmp(plottype, 'bar'),
%    cor = defcolor;
%end

%%


%use 2 levels even for dim 1
% if dimm == 1,
%     if length(levels) == length(Ymean),
%         levels =  [{''} levels];
%     else
%         levels = {[], {}};
%     end
% end



%% open figure
% if strcmp(plottype, 'noplot')
%     plothandle = [];
%     return;
% end

% !! what to do with this
if strcmp(doclf, 'clf')
    switch dimm
        case {1 2}
            myfig(name, varnames{1}, y_label, [], [], 0, 0, 0);
        case 3
            myfig(name, varnames{1}, y_label, [], [], 0, 0, 0, 250*length(levels{3}), 200);
        case 4
            myfig(name, varnames{1}, y_label, [], [], 0, 0, 0, 250*length(levels{3}), 200*length(levels{4}));
    end
    clf;
end

if strcmp(plottype,'imagesc') && dimm>2
    clim = [min(M(:)) max(M(:))];
end

% whether figure is currently on hold
is_hold = ishold;
hold on;

%% plot
switch dimm %number of dimension
    case 1  % just one curve/bar series
        setclim(clim); titl = '';
        
        plothandle = pcurve(X(:), M(:), L(:), U(:), pval, [varnames {''}], levels);  %works without ' for curve/errorbars (maybe depends also if Ymean is row or column)
        
        % legend off;
        
    case 2  % multiple curve/bars in same panel
        setclim(clim); titl = '';
        plothandle = pcurve(X, M, L, U, pval,   varnames, levels);
        
    case 3 % different panels
        for s = 1:siz(3)
            subplot2(siz(3),s);
            setclim(clim);
            if ~isempty(levels{3})
                titl = shortlabel(varnames{3}, levels{3}{s}); %title
            end
            ph = pcurve(X, M(:,:,s), L(:,:,s), U(:,:,s),pval(:,:,s), varnames(1:2), levels(1:2));
            % if ~all(isnan(Yerr(:))) && all(all(isnan(Yerr(:,:,s)))),
            %     ph.error = [];
            % end
            plothandle(s) = ph;
            if s>1
                legend(gca,'off');
            end
            axis tight;
        end
        sameaxis;
        
    case 4 % different panels
        for s1 = 1:siz(3)
            for s2 = 1:siz(4)
                subplot(siz(4), siz(3), s1 + siz(3)*(s2-1));
                setclim(clim)
                titl = [shortlabel(varnames{3}, levels{3}{s1}) ', ' shortlabel(varnames{4}, levels{4}{s2})];
                
                % plot
                plothandle(:,s1,s2) = pcurve(X, M(:,:,s1,s2), L(:,:,s1,s2), U(:,:,s1,s2), pval(:,:,s1, s2), ...
                    varnames(1:2), levels(1:2));
                if s1>1 || s2>1
                    legend off;
                end
            end
            axis tight;
        end
        sameaxis;
end

if ~is_hold
    hold off;
end

warning(imaginary_warning.state,'MATLAB:plot:IgnoreImaginaryXYPart'); % turn warning for imaginary data back to previous value

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% SUBFUNCTIONS %%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% plotting subfunction
    function phandle = pcurve(X, MM, LL, UU, PP, vnames, labelz)
        
        %         %abscissa values !! deleted because it's already taken care of
        %         above
        %         if isempty(X),
        %             % if isvector(Ymean),
        %             %     X = 1:length(Ymean);
        %             % else
        %             try %!!!! should not be here, and seems this is already aounrd line 380
        %                 X = cellfun(@str2double, labelz{1});  %convert from llevels (if numeric)
        %                 if any(isnan(X)) || isempty(X),
        %
        %                     X = 1:size(MM,1);
        %                 end
        %             catch%(otherwise simply integers starting from one)
        %                 X = 1:size(MM,1);
        %             end
        %
        %
        %             % end
        %         end
        X = X(:)';
        
        hold on;
        
        %% compute probability values to display
        %         if isnumeric(PP),
        %             if length(PP)==1,
        %                 PP = struct('FF1', Inf(1,nbv), 'FF2', Inf(1,nbp));  %dont' draw any signifchar
        %             else
        %                 PP = struct('FF1', PP, 'FF2', Inf);
        %             end
        %         else
        %             try
        %                 PP.FF1 = PP.(vnames{1});
        %             catch
        %                 warning('wu:probdisplay', 'prob displaying did not work');
        %             end
        %             try
        %                 PP.FF2 = PP.(vnames{2});
        %             catch
        %                 warning('wu:probdisplay', 'prob displaying did not work');
        %             end
        %         end
        %         if nbv == 2, %de-squeeze
        %             PP.FF2 = reshape(PP.FF2, [1 1 nbp]);
        %         end
        %         if nbp == 2,
        %             PP.FF1 = reshape(PP.FF1, [1 1 nbv]);
        %         end
        
        %% lables for legend
        add_legend = length(labelz)>1  && ~isempty(labelz{2}) && ~all(cellfun(@isempty, labelz{2})); %except when only one curve and no attached label
        if  add_legend
            legend_labels = labelz{2};
            %legend_labels = cell(1, nbv);
            %for w = 1:nbv
            % forlegend{w} = shortlabel(factoz{3}, labelz{2}{w});
            %   legend_labels{w} = labelz{2}{w};
            %end
        end
        
        
        %%%%%%%%%  BARS %%%%%%
        switch plottype
            case 'bar'
                
                phandle = mybar(X, MM, LL, UU, 'facecolor', cor, errorbarpars{:});
                
                %                 %display probability values
                %                 if nbv ==2 && all(~isinf(PP.FF2))  % same x-values
                %                     phandle.signif = [];
                %                     for p=1:nbp
                %                         if PP.FF2(1,1,p)<.1,
                %                             if (any(MM(p,:)>0)),
                %                                 hi = max(MM(p,:)+1.1*EE(p,:));
                %                             else
                %                                 hi = min(MM(p,:) - 1.1*EE(p,:));
                %                             end
                %                             phandle.pvalue(p) = text(p, hi , signifchar(PP.FF2(1,1,p)), 'FontSize', 24, 'HorizontalAlignment', 'center');
                %                         end
                %                     end
                %                 end
                %                 if nbp==2 && all(~isinf(PP.FF1))  % between x-values
                %                     phandle.pvalue=[];
                %                     whichcor = 1 + round((size(cor,1)-1) * (0:nbv-1)/(nbv-1));
                %                     lightcolour = cor(whichcor,:);
                %                     for w=1:nbv
                %                         if PP.FF1(1,1,w)<.1
                %                             phandle.pvalue(p) = text(1.5, max(MM(:,p)), signifchar(PP.FF1(1,1,w)), 'Color', lightcolour(w,:), ...
                %                                 'FontSize', 24, 'HorizontalAlignment', 'center');
                %                         end
                %                     end
                %                 end
                
                if add_legend
                    if strcmpi(legend_style, 'standard')
                        hleg = legend(legend_labels{:}, 'Location', 'Best');
                    elseif strcmpi(legend_style, 'color')
                        hleg = legend_color(legend_labels{:}, 'Location', 'Best');
                        
                    end
                end
                if ~isempty(vnames{2})
                    hlegtitle = get(hleg,'title');
                    set(hlegtitle,'string',vnames{2});
                    set(hlegtitle, 'fontsize',get(gca, 'fontsize'));
                    set(hlegtitle, 'Position', [0.1 1.05304 1]);
                    set(hlegtitle, 'HorizontalAlignment', 'left');
                end
                %legend boxoff
                
                
                %% %%%% LINES %%%%%%%%%
            case 'curve'
                %                 if nbp==2 && all(~isinf(PP.FF1)),
                %                     phandle.pvalue = [];
                %                 end
                
                
                % 'fill' option: compute vector for surface
                if any(strcmp(errorstyle, {'fill', 'area', 'marge'}))
                    Xfill = [X X(end:-1:1)];                                                %abscissa for the surface
                    Yfill = [ MM-LL ; flipud(MM+UU) ]; % ordinates for the surface
                end
                
                % plot errors
                if ~all(isnan(LL(:))) || ~all(isnan(UU(:)))
                    for w=1:nbv  % corresponding to each different curve
                        switch lower(errorstyle)
                            case {'bar', 'bars', 'errorbar', 'errorbars'} %error bars
                                if strcmp(maxis, 'y') % vertical bars
                                    phandle.E(:,w) = myerrorbar(X, MM(:,w), LL(:,w), UU(:,w),'vertical', 'color', cor(w,:), errorbarpars{:},'none');
                                else % horizontal bars
                                    phandle.E(:,w) = myerrorbar(MM(:,w), X,  LL(:,w), UU(:,w),'horizontal', 'color', cor(w,:), errorbarpars{:},'none');
                                end
                                
                            case  {'fill', 'area', 'marge'} %error areas
                                nonnan = ~isnan(Yfill(:,w));
                                if any(nonnan)
                                    phandle.E(:,w) = fill( Xfill(nonnan), Yfill(nonnan,w), lightcolour(w,:), 'LineStyle', errorlinestyle);
                                else
                                    phandle.E(:,w) = nan;
                                end
                            case {'curve', 'curves', 'errorcurve', 'errorcurves'} %error lines
                                if strcmp(maxis, 'y')
                                    phandle.E(:,w) = plot( X, [MM(:,w)-LL(:,w) MM(:,w)+UU(:,w)], 'LineWidth', linewidth/2, 'Color', cor(w,:), 'LineStyle', errorlinestyle );
                                else
                                    phandle.E(:,w) = plot(  [MM(:,w)-LL(:,w) MM(:,w)+UU(:,w)], X,'LineWidth', linewidth/2, 'Color', cor(w,:), 'LineStyle', errorlinestyle );
                                end
                        end
                    end
                end
                
                for w = 1:nbv % for each variable
                    
                    %plot curve
                    if strcmp(maxis, 'y')
                        phandle.M(w) = plot(X, MM(:,w), 'Color', cor(w,:), 'Marker',marker, 'markersize', markersize, ...
                            'Linestyle', linestyle, 'linewidth', linewidth);
                    else
                        phandle.M(w) = plot( MM(:,w), X,'Color', cor(w,:), 'Marker',marker, 'markersize', markersize, ...
                            'Linestyle', linestyle, 'linewidth', linewidth);
                    end
                    
                    %                     %plot significances character between 2 x-values
                    %                     if nbp==2 && PP.FF1(1,1,w)<.1
                    %                         phandle.signif(w) = text(mean(X(1:2)), max(MM(:,w)), signifchar(PP.FF1(1,1,w)), 'FontSize', 24, ...
                    %                             'Color', cor(w,:), 'HorizontalAlignment', 'center');
                    %                     end
                    %
                    %                     if nbp==2 && PP.FF1(1,1,w)<.1,    % add significance character to legend label
                    %                         labelz{2}{w} = [labelz{2}{w} '(' signifchar(PP.FF1(1,1,w)) ')' ];
                    %                     end
                end
                
                % add legend
                if add_legend
                    if strcmp(legend_style, 'standard')
                        legend(phandle.M, legend_labels, 'Location', 'Best');
                    else
                        legend_color(phandle.M, legend_labels); %, 'Location', 'Best');
                    end
                end
                
                %%%%%%%%%% XY TYPE
            case 'xy'
                
                %add vertical and horizontal error bars
                %  barwidd = ticklength;
                phandle.E(:,1) = myerrorbar(MM(1,:), MM(2,:), LL(2,:),UU(2,:), 'vertical','color',cor(1,:), errorbarpars{:},'linestyle',linestyle);
                phandle.E(:,2) = myerrorbar(MM(1,:), MM(2,:), LL(1,:),UU(1,:), 'horizontal', 'color',cor(1,:), errorbarpars{:},'linestyle',linestyle);
                
                %horizontal error bar (add bar width property)
                %                 dem = herrorbar(MM(1,:), MM(2,:), EE(1,:), EE(1,:));
                %                 delete(dem(2)); %remove the line
                %                 set(dem(1), 'Color', cor(1,:));
                %                 phandle.E(2) = dem(1);
                
                %plot curve
                phandle.M = plot(MM(1,:), MM(2,:), 'Color', cor(1,:), 'Marker',marker, ...
                    'MarkerSize', markersize, 'Linestyle', linestyle, 'linewidth', linewidth);
                
                % plot identity line
                if IdentityLine
                    xl = xlim; yl = ylim;
                    xx = [min(xl(1),yl(1)) max(xl(2),yl(2))];
                    phandle.identityline = plot(xx,xx,'color',.5*[1 1 1]);
                    uistack(phandle.identityline, 'bottom');
                end
                
                %%%%%%%%%% IMAGESC TYPE
            case 'imagesc'
                ylabl = cellfun(@str2double, labelz{2});
                if any(isnan(ylabl)) || isempty(ylabl)
                    ylabl = [1 nbv];
                end
                phandle = imagesc(X, ylabl, MM');
                axis tight;
        end
        
        %% plot significance lines
        if ~all(isnan(PP(:)))
            [Xord, i_ord] = sort(X); % sort values along X bar to get contiguous
            if length(X)>1
                Xord = [2*Xord(1)-Xord(2) Xord 2*Xord(end)-Xord(end-1)]; % extrapolate for point before first and point after last
            else % if just one point
                Xord = Xord + (-1:1);
            end
            for w=1:nbv % for each line/bar series
                sig = PP(:,w)'<threshold; % values that reach significance
                sigord = [false sig(i_ord) false]; % re-order according to increasing X values, and add false values for inexisintg points 0 and end+1
                Ysig = nan(1,length(Xord)); % Y values for points is nan (do not draw) by default
                Ysig(sigord) = 1; % and non-nan only for significant values
                Xsig = Xord;
                singlepoints = 1 + find(sigord(2:end-1) & ~sigord(1:end-2) & ~sigord(3:end)); % significant points between two non significant points
                for ss = fliplr(singlepoints) % for each single oiunt (we go in reverse order to avoid confusion between indices as size changes
                    Xsig = [Xsig(1:ss-1) Xsig(ss)-.1*diff(Xsig(ss-1:ss)) Xsig(ss)+.1*diff(Xsig(ss:ss+1)) Xsig(ss+1:end)]; % create a short segment around that point
                    Ysig = [Ysig(1:ss-1) 1                               1                               Ysig(ss+1:end)];
                end
                
                % plot the line
                if strcmp(maxis, 'y')
                    phandle.S = plot(Xsig, y_pval(w)*Ysig, 'color', cor(w,:));
                else
                     phandle.S = plot(y_pval(w)*Ysig, Xsig,  'color', cor(w,:));
                end
                set( phandle.S, 'Tag', 'Significant');
            end
        end
        if isfield(phandle, 'M')
              set( phandle.M, 'Tag', 'mean');
        end
        if isfield(phandle, 'E')
         %     set( phandle.E, 'Tag', 'error');
        end
        
        %% axis labels and tick labels
        if strcmp(plottype, 'xy')
            if ~isempty(labelz{1}) && ~isempty(labelz{1}{1}) && ~isequal(str2double(labelz{1}{1}),1)
                xlabel( labelz{1}{1});
            end
            if length(labelz{1})>1 && ~isempty(labelz{1}{2}) && ~isequal(str2double(labelz{1}{2}),2)
                xlabel( labelz{1}{2});
            end
        else
            if ~isempty(y_label)    % add label for dependent variable
                if strcmp(maxis, 'y')
                    ylabel(y_label);
                else
                    xlabel(y_label);
                end
            end
            if ~isempty(vnames{1})      %add label to indep variable axis
                if strcmp(maxis, 'y')
                    xlabel(vnames{1});
                else
                    ylabel(vnames{1});
                end
            end
            
            %% place labels for x axis
            if VerticalLabel % if vertical labels over bars
                y_vertlabels =  MM + 1.1*UU;
                y_vertlabels(isnan(UU)) = MM(isnan(UU)) + .05*diff(eval([faxis 'lim']));
                if nbp>1 && nbv>1
                    y_vertlabels = nanmax(y_vertlabels,[],2);
                end
                
                if nbp>1
                
                for p=1:nbp
                    if strcmp(maxis, 'y')
                     phandle.label(p) = text(X(p),y_vertlabels(p), labelz{1}(p), 'rotation',90);
                    else
                        phandle.label(p) = text(y_vertlabels(p), X(p), labelz{1}(p));
                    end
                end
                 
                else % bars
                    legend off;
                    for vv=1:nbv
                    if verLessThan('matlab', '8.4')
                R_eb =get(get(phandle.M(vv),'children'),ref_field);
                this_x = mean(R_eb([1 3],1));
            else
               this_x = X(1) + phandle.bar(vv).([upper(faxis) 'Offset']);
                    end
                    if strcmp(maxis, 'y')
                     phandle.label(vv) = text(this_x,y_vertlabels(vv), labelz{2}(vv), 'rotation',90);
                    else
                        phandle.label(vv) = text(y_vertlabels(vv), this_x, labelz{2}(vv));
                    end
                    
                    end
                end
                
            elseif (~isempty(labelz{1}) && ~isequal(num2strcell(X), labelz{1}(:)')) || nbp < 10 %length(labelz{2})<10,
                [xtickval,xorder] = unique(X);
                set(gca, [faxis 'Tick'], unique(xtickval));
                if ~isempty(labelz{1})
                    set(gca, [faxis 'TickLabel'], labelz{1}(xorder));
                end
            end
            
            % rotate xtick labels by specified angle
            if xtickangle ~=0
                if verLessThan('matlab', '8.4') % use user-supplied function
                    rotateXLabels( gca, xtickangle);
                else   % use built-in property
                    set(gca, 'XTickLabelRotation', xtickangle);
                end
                
            end
                  
        end
        
        %add title
        if isfield(PP, 'interaction')
            title([titl ' p=' num2str(PP.interaction)]);
        elseif ~isempty(titl)
            title(titl);
        end
    end

%%%%%%%%%%%%%
% convert color argument into n-by-3 RGB matrix
    function   C = color2mat(C, nbv)
        if ischar(C)
            switch lower(C)
                case  {'jet', 'gray', 'pink', 'hsv', 'hot', 'cool', 'copper', 'flag', 'prism'},
                    C = eval(C);
                    if nbv>1
                        corvec = 1 + (size(C,1)-1) * (0:nbv-1)/(nbv-1);
                    else
                        corvec = 1;
                    end
                    C = C(floor(corvec),:);
                case {'flat','colormap'} %interpolate colors from colormap
                    C = colormap;  %current colormap
                    if nbv>1
                        corvec = 1 + (size(C,1)-1) * (0:nbv-1)/(nbv-1);
                    else
                        corvec = 1;
                    end
                    C = C(floor(corvec),:);
                otherwise  %just one colour symbol (e.g. 'k')
                    C = {C};
            end
        end
        
        % if cell array, convert to matrix of RGB values
        if iscell(C)
            cor_mat = zeros(length(C),3);
            for c=1:length(C)
                if isnumeric(C{c}) % RGB value
                    cor_mat(c,:) = C{c};
                elseif ischar(C{c}) % letter (e.g. 'k', 'b', etc.)
                    cor_mat(c,:) = rem(floor((strfind('kbgcrmyw', C{c}) - 1) * [0.25 0.5 1]), 2);
                else
                    error('incorrect value for colour: should be vector of RGB value or single character');
                end
            end
            C = cor_mat;
        end
    end
%%%%

    function str = shortlabel(fname, vname)
        if isempty(str2double(vname)) || isempty(fname)
            str = vname;
        else
            str = [fname '=' vname];
        end
    end

%% set colors limit
    function setclim(clim)
        if isequal(clim, 'auto')
            set(gca, 'CLimmode', 'auto');
        else
            set(gca,'Clim',clim);
        end
    end

%%default color
function C = defaultcolor()
 C = [ ...
         0     0     0  ; ...   %black 
         1      0     0  ; ... %red
        0     0     1  ; ...  % blue
         0    .5     0  ; ...
      0.25   .25   .25  ; ...
      0.75   .75     0  ; ...
      0.75     0   .75  ; ...
         0   .75   .75  ; ...
         1     0     0  ; ...
         0    .5     0  ; ...
         0     0     1 ];
    
  C = num2cell(C,2);
end

end

