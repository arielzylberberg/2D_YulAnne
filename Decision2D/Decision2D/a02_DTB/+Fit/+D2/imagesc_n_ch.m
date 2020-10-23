function p = imagesc_n_ch(p0, varargin)
% p = imagesc_n_ch(p0, varargin)
%
% p0(t, cond1, cond2, ch1, ch2)
% p(cond1 x ch1, cond2 x ch2)

%%
if ndims(p0) == 5
    siz0 = size(p0);
    n_cond = siz0(2:3);
    p1 = permute(sum(p0, 1), [4, 2, 5, 3, 1]);
elseif ndims(p0) == 4
    siz0 = size(p0);
    n_cond = siz0(1:2);
    p1 = permute(p0, [3, 1, 4, 2]);
else
    error('ndims(p0) must be 4 or 5!');
end
p = reshape(p1, n_cond * 2);

y = (0.5:0.5:n_cond(1)) +.25;
x = (0.5:0.5:n_cond(2)) +.25;

imagesc(x, y, p');
axis xy
xlabel('Cond M');
ylabel('Cond C');

grid off;
set(gca, ...
    'XTick', 1:n_cond(1), ...
    'YTick', 1:n_cond(2), ...
    'TickLength', [0 0]);

h1 = crossLine('h', 0.5:(n_cond(1)+0.5), 'k-');
uistack(h1, 'top');

h1 = crossLine('v', 0.5:(n_cond(2)+0.5), 'k-');
uistack(h1, 'top');

axis square;