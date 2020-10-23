function feat = determine_feat(ax)
% Determine feat from points' maximum x coordinate
    if ~exist('ax', 'var'), ax = gca; end

    xy = bml.plot.get_all_xy(ax);
    x = xy(:,1);
    
    if any(x == 0.128)
        feat = 'M';
    else
        feat = 'C';
    end
end