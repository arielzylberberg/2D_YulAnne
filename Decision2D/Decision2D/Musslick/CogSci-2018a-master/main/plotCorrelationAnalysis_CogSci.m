function plotCorrelationAnalysis_CogSci(std_values, sim_log, var_label)

plotsettings;

y = nanmean(sim_log, 2);
% sem = std(sim_log, [], 2) ./ sqrt(y);

fig = figure(1);
set(fig, 'Position', [100, 100 200 200]);
plot(std_values, y, 'LineWidth', plotSettings.lineWidth, 'Color', plotSettings.colors(cSingle, :));
hold on;

points_x = repmat(std_values, size(sim_log, 2), 1);
points_y = sim_log';

scatter(points_x(:), points_y(:), 'x');
xlabel({'Standard Deviation of ', var_label}, 'FontSize', plotSettings.xLabelFontSize-3); 
ylabel({'Correlation Between True and', 'Estimated Control Costs'}, 'FontSize', plotSettings.yLabelFontSize-3);
set(gca, 'FontSize', plotSettings.gcaFontSize-3);
ylim([0 1]);

end