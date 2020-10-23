function h = plot_tda(p_rt, varargin)
S = varargin2S(varargin, {
    'dim', 1
    'to_plot_shade', true
    'linewidth', 1
    });
dim = S.dim;

if dim == 2
    p_rt = permute(p_rt, [1, 3, 2, 5, 4]);
end

W = Fit.D2.Common.Main;
t = W.t(:);

m = 0.3;
s = 0.15;
filt = gampdf_ms(t, m, s);
p_td = bml.math.conv_t_back(p_rt, filt);
        
%%
p_td_fold = (p_td ...
    + flip(flip(p_td, 2), 4) ...
    + flip(flip(p_td, 3), 5) ...
    + flip(flip(flip(flip(p_td, 2), 3), 4), 5)) / 4;
p_td_fold = p_td_fold(:, round(end/2):end, round(end/2):end, :, :);

dif_irr_incl = 2:3;
p_td_slice = squeeze(sum(p_td_fold(:, dif_irr_incl, :, :, 2), 2));
n_tr_time = sum(p_td_slice, 3);
p_accu = p_td_slice(:,:,2) ./ n_tr_time;

nt = length(t);
e_accu = zeros(size(p_accu));

irr_groups = {1:2, 3:4, 5};
n_dif_irr = numel(irr_groups);
colors = hsv2rev(n_dif_irr);

%%
h = cell(1, n_dif_irr);
for i_irr = 1:n_dif_irr
    irr = irr_groups{i_irr};

    phat = zeros(nt, 1);
    ci = zeros(nt, 2);
    
    p1 = squeeze(sum(p_td_slice(:, irr, :), 2));
    p_cumsum1 = cumsum(sum(p1, 2) / sum(p1(:)));
    t_incl1 = (p_cumsum1 >= 0.05) & (p_cumsum1 <= 0.95);
    
    for t1 = 1:nt
        p11 = p1(t1,:);
        
        phat(t1) = (p11(2) + 1) / sum(p11 + 1); % p_accu(t1,irr);
        a = p11(2) + 1;
        b = p11(1) + 1;
        ci(t1,:) = betainv([0.25, 0.75], a, b);
        
%         [phat(t1), ci(t1,:)] = binofit( ...
%             p_td_slice(t1,irr,2), n_tr_time(t1, irr));
    end
    ci = bsxfun(@minus, ci, phat);
    
    color = colors(i_irr,:);
    t1 = t(t_incl1);
    y1 = phat(t_incl1);
    ci1 = ci(t_incl1,:);
    if S.to_plot_shade
        h{i_irr} = errorbarShade(t1, y1, ci1, ...
            color, [], {'LineWidth', S.linewidth'});
    else
        h{i_irr} = plot(t1, y1, 'Color', color, 'LineWidth', S.linewidth);
    end
    ylim([0.5, 1]);
    bml.plot.beautify;    
end
end