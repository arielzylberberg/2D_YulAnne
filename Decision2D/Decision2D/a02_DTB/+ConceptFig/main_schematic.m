%% Schematic
init_path;
Pl = ConceptFig.PlotEvAccum;
clf;
h = Pl.plot_accum( ...
    'plot_arrow', false, ...
    'height_ch', 0.5);
delete(h.text);
axis(gca, 'auto')
ylabel('');
xlabel('');
bml.plot.position_subplots(gca, ...
    'margin_left', 0.01, ...
    'margin_bottom', 0.01, ...
    'margin_right', 0.01, ...
    'margin_top', 0.01);
savefigs('../Data_2D/ConceptFig.main_schematic/dtb_schematic', 'size', [200, 150]);

%% Figure
fig_tag('schematic');
clf;
n_row = 1;
n_col = 5;
ax = subplotRCs(n_row, n_col);

slope_easy = 2;
slope_hard = 0.5;

colors = bml.plot.colormaps.cool2(2);
color_easy = colors(1,:); % % bml.plot.color_lines('b');
color_hard = colors(2,:); % bml.plot.color_lines('r');
% color_both = bml.plot.color_lines('g');

xy0_easy = [0, 2.1];
xy0_hard = [0, 0];
y_lim = [-0.6, 2.1 + 1.6];

%%
bml.plot.position_subplots(ax, ...
    'col_rel', [3, 2.4, 3.6, 5.8, 2.5], ...
    ... 'row_rel', [2, 1, 1], ...
    'btw_row', 0.01, ...
    'btw_col', 0.01, ...
    'margin_left', 0.03, ...
    'margin_right', 0.01, ...
    'margin_top', 0.08, ...
    'margin_bottom', 0.01);

%% Evidence Accumulation process
axes(ax(1,1));
cla;
Pl.plot_accum;

%% Parallel
axes(ax(1,2));
cla;
Pl.xy0 = xy0_easy;
[t0e, t1e] = Pl.plot_parallel(slope_easy, color_easy);
hold on;
title('Parallel');

% axes(ax(2,2));
Pl.xy0 = xy0_hard;
[t0h, t1h] = Pl.plot_parallel(slope_hard, color_hard);

% hold on;
% h = plot([t1e, t1e], [0, xy0_easy(2) + 1], ':', ...
%     'Color', color_easy * 0.2 + 0.8, 'LineWidth', 1);
% uistack(h, 'bottom');
% 
% h = plot([t0e, t0e], [0, xy0_easy(2) + 1.2], ':', ...
%     'Color', [0, 0, 0] * 0.2 + 0.8, 'LineWidth', 1);
% uistack(h, 'bottom');
% 
% h = plot([t1h, t1h], [0, xy0_hard(2) + 1], ':', ...
%     'Color',  color_hard * 0.2 + 0.8, 'LineWidth', 1);
% uistack(h, 'bottom');
% 
% C = varargin2C({
%     'Length', 6
%     'TipAngle', 20
%     'LineWidth', 1
%     'Color', color_hard
%     }, Pl.arrow_style);
% arrow([t1e, 0], [t1h, 0], C{:});
% hold on;
% 
% plot(t0e, 0, 'o', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', ...
%     'MarkerSize', 8);
% plot(t1e, 0, 'o', 'MarkerFaceColor', color_easy, 'MarkerEdgeColor', 'none');
% hold off;
ylim(y_lim);

%% Serial
axes(ax(1,3));
cla;
Pl.xy0 = xy0_easy;
[t0e, t1e] = Pl.plot_serial(slope_easy, color_easy);
title('Serial');
hold on;

% axes(ax(2,3));
% cla;
Pl.xy0 = xy0_hard;
[t0h, t1h] = Pl.plot_serial(slope_hard, color_hard);

% hold on;
% h = plot([t1e, t1e], [0, xy0_easy(2) + 1], ':', ...
%     'Color', color_easy * 0.2 + 0.8, 'LineWidth', 1);
% uistack(h, 'bottom');
% 
% % h = plot([t0e, t0e], [0, xy0_easy(2) + 1.2], ':', ...
% %     'Color', [0, 0, 0] * 0.5 + 0.5, 'LineWidth', 1);
% % uistack(h, 'bottom');
% 
% h = plot([t1h, t1h], [0, xy0_hard(2) + 1], ':', ...
%     'Color',  color_hard * 0.2 + 0.8, 'LineWidth', 1);
% uistack(h, 'bottom');
% 
% C = varargin2C({
%     'Length', 6
%     'TipAngle', 20
%     'LineWidth', 1
%     'Color', color_hard
%     }, Pl.arrow_style);
% arrow([t1e, 0], [t1h, 0], C{:});
% hold on;
% plot(t1e, 0, 'o', 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k', ...
%     'MarkerSize', 8);
% plot(t1e, 0, 'o', 'MarkerFaceColor', color_easy, 'MarkerEdgeColor', 'none');
% 
% % y_lim = ylim;
% % h = plot([t1e, t1e], [0, y_lim(2)], ':', ...
% %     'LineWidth', 2, ...
% %     'Color', color_easy * 0.5 + 0.5);
% % uistack(h, 'bottom');
ylim(y_lim);

%% Serial w/ switching
axes(ax(1,4));
cla;
Pl.xy0 = xy0_easy;
Pl.plot_serial_switch(slope_easy, color_easy);
title(sprintf('Serial + Switching'));
hold on;

% axes(ax(2,4));
% cla;
Pl.xy0 = xy0_hard;
Pl.plot_serial_switch(slope_hard, color_hard);
ylim(y_lim);

%% Targetwise
axes(ax(1,5));
cla;
ylim(y_lim);
drawnow;
Pl.xy0 = xy0_easy;
Pl.plot_targetwise(slope_easy, color_easy);
title('Targtwise');
hold on;

% axes(ax(2,5));
Pl.xy0 = xy0_hard;
Pl.plot_targetwise(slope_hard, color_hard);

%% Save
bml.plot.savefigs('../Data_2D/ConceptFig.main_schematic/dtb_models_scheme', ...
    'size', [550, 150]);



