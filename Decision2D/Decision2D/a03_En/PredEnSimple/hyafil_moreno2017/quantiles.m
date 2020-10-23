function [Z, B] = quantiles(Y, n, G, subset)
%[Z quantvalues] = quantiles(Y, n, ) bins Y values into n quantiles and
%assigns in Z the corresponding quantile of each value of Y
%
%%[Z quantvalues] = quantiles(Y, n, G, [,subset]) to group along values of
%%G
%
%also [Z quantvalues] = quantiles(Y, prop, G [,subset]) where prop is a vector of values
%btw 0 and 1


if nargin>=4,
    Yin = Y(:,subset);
    Gin = G(:,subset);
else
    if nargin<3,
        G = ones(size(Y));
    end
    Yin = Y;
    Gin = G;
    subset = 1:length(Y);
end

%turn into edges vector if required
if length(n)==1 && n>=1
    prop = 0: 1/n :1;
    prop(end) =[];
else prop =n;   %already a vector
end

if isempty(Yin),
    Z = Yin;
    return;
end

for s=1:size(G,1)
    %  Gin(s,:) = replace(Gin(s,:));
    Gin(s,:) = grp2idx(Gin(s,:));
end

%group data if required
if nargin>=3 && ~isempty(Gin),
    A = grpn(Yin, Gin);
    returncell = 1;
else
    A = {Yin};
    returncell = 0;
end

%compute quantiles
B = cell(size(A));
for a=1:numel(A)
    if ~isempty(A{a})
        B{a} = quantile(A{a}, prop);
    else
        B{a} = NaN;
    end
end

%assign to each value of Y the associate quantile number
Zin = nan(size(Yin));
for i=1:length(Yin)
    if ~isnan(Yin(i))
        coords = num2cell(Gin(:,i)');
        Zin(i) = sum(Yin(i)>= B{coords{:}});
        if Zin(i)==0
            error('problem de script?')
        end
    end
end

Z = -ones(size(Y));
Z(subset) = Zin;

if ~returncell,
    B = B{1};
end