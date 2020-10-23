Pl = ConceptFig.PlotEvAccum;

%% Cartoon predictions
fig_tag('cartoon_pred');
colors = bml.plot.colormaps.cool2(2);
color_easy = colors(1,:); % % bml.plot.color_lines('b');
color_hard = colors(2,:); % bml.plot.color_lines('r');
% color_easy = bml.plot.color_lines('b');
% color_hard = bml.plot.color_lines('r');

clf;
n_row = 4;
n_col = 2;
ax = subplotRCs(n_row, n_col);
bml.plot.position_subplots(ax, ...
    'btw_row', [0.02, 0.06, 0.06], ...
    'btw_col', 0.1, ...
    'margin_top', 0.03, ...
    'margin_left', 0.18, ...
    'margin_right', 0.05, ...
    'margin_bottom', 0.07, ...
    'col_rel', [2, 1]);

%% Serial + Switching RT-cond
col = 3;
ax1 = ax(col,1);
axes(ax1);
cla;
y_easy_max = 1.5;
y_max = 3; % (1.7 + 1) * 1.1;
x_lim = [-2.2, 2.2];
% f_xlim = @(col) [-2.2, -2.2 + 4.4 * diff(ax(1,col).XLim)/diff(ax(1,2).XLim)];
% f_rt_xlim = @(col) [0, y_max * diff(ax(1,col).XLim)/diff(ax(1,2).XLim)];

ylim([0, y_max]);
xlim(x_lim);
% xlim(f_xlim(col));
[ys_easy, ys_hard] = Pl.plot_rt_cond(1.7);
xlim(x_lim);
title('Serial + switching');
axis on
set(gca, 'XTick', [-2, 0, 2], 'XTickLabel', [], 'YTick', 0:1.5:3, 'YTickLabel', []);
%     'XTickLabel', {'Easy left', 'Ambiguous', 'Easy right'});
bml.plot.beautify;
ylabel('RT');

%% Serial + Switching RT-RT
axes(ax(col,2));
cla;
axis equal
ylim([0, y_max]);
xlim([0, y_easy_max]);
% xlim(f_rt_xlim(col));
drawnow;
Pl.plot_rt_rt(ys_easy, ys_hard);
ylim([0, y_max]);

axis on;
bml.plot.beautify;
set(gca, 'XTickLabel', [], 'YTickLabel', [], 'XTick', 0:1.5:3, 'YTick', 0:1.5:3);
xlabel('RT_{strong C}');
ylabel('RT_{weak C}');
text(0.1, max(ys_hard)+0.4, 'Slope > 1');

%% Serial RT-cond
col = 2;
% axes(ax(1,col));
axes(ax(col,1));
cla;
ylim([0, y_max]);
xlim(x_lim);
drawnow;
[ys_easy, ys_hard] = Pl.plot_rt_cond(1);
xlim(x_lim);
ylim([0, y_max]);
text(0, 2.5, 'Serial', 'FontSize', 11, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center');
% title('Serial');

%% Serial RT-RT
% axes(ax(2,col));
axes(ax(col,2));
cla;
axis equal
ylim([0, y_max]);
xlim([0, y_easy_max]);
drawnow;
Pl.plot_rt_rt(ys_easy, ys_hard);
ylim([0, y_max]);
text(0.1, max(ys_hard)+0.3, 'Slope = 1');

%% Parallel RT-cond
col = 1;
axes(ax(col,1));
% axes(ax(1,col));
cla;
ylim([0, y_max]);
xlim(x_lim);
[ys_easy, ys_hard, he, hh] = Pl.plot_rt_cond(0.5);
ylim([0, y_max]);
xlim(x_lim);
title('Parallel');
ylim([0, y_max]);
legend([he, hh], {'Easy color', 'Hard color'}, ...
    'Location', 'North');

%% Parallel RT-RT
% axes(ax(2,col));
axes(ax(col,2));
cla;
axis equal
ylim([0, y_max]);
xlim([0, y_easy_max]);
drawnow;
Pl.plot_rt_rt(ys_easy, ys_hard, 'LineStyle', '--');
ylim([0, y_max]);
text(0.1, max(ys_hard)+0.3, 'Slope < 1');

%% Targetwise ch-cond
col = 4;
% axes(ax(1,col));
axes(ax(col,1));
cla;
hh = Pl.plot_ch_cond(3, color_hard);
hold on;
he = Pl.plot_ch_cond(1, color_easy);
ylim([-0.05, 1.05]);
xlim(x_lim);
axis on;
bml.plot.beautify;
set(gca, 'XTick', [-2, 0, 2], 'YTick', [0, 0.5, 1], ...
    'XTickLabel', {'Strong left', 'Weak', 'Strong right'});
ylabel('P_{right}');
xlabel('Motion strength');
title('Targetwise');

axes(ax(col,2));
% axes(ax(2,col));
cla;
axis off;

%% Save
bml.plot.savefigs('../Data_2D/ConceptFig.main_schematic/dtb_models_pred', ...
    'size', [250, 500]);
