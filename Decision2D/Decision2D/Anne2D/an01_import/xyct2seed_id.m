function id = xyct2seed_id(xyct)
% Gives a seed ID given xyct.
id = abs(xyct(1,1)) * 1e6 + abs(xyct(1,2));
end