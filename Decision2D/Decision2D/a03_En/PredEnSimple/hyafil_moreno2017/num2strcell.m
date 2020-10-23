function str = num2strcell( vect, ndig)
%str = num2strcell( vect)
%str = num2strcell(vect, ndigits) to ensure a minimum value of digits
%
%str = num2strcell(string, vect) to insert numbers into a given string
%eg num2strcell('X%d', 2:5)


if isnumeric(vect)

str = cell(size(vect));
for i=1:numel(vect),
    str{i} = num2str(vect(i));
    if nargin>1 && length(str{i})<ndig
        zer = repmat('0', 1, ndig - length(str{i}));
        str{i} = [zer str{i}];
    end
end

else
    istr = vect;
    vect = ndig;
    str = cell(size(vect));
    for i=1:numel(vect),
        str{i} = sprintf(istr, vect(i));
    end
end