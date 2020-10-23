function plotBetweenExperimentAnalysis_CogSci(std_values, sim_log, var_label)

plotsettings;

y = nanmean(sim_log, 2);
% sem = std(sim_log, [], 2) ./ sqrt(y);

fig = figure(1);
set(fig, 'Position', [100, 100 220 200]);
plot(std_values, y, 'LineWidth', plotSettings.lineWidth, 'Color', plotSettings.colors(cSingle, :));
hold on;

points_x = repmat(std_values, size(sim_log, 2), 1);
points_y = sim_log';

scatter(points_x(:), points_y(:), 'x');
xlabel({['Correlation of ' num2str(var_label)], 'Between Experiments'}, 'FontSize', plotSettings.xLabelFontSize-3.5); 
ylabel({'Correlation of Control Costs', 'Between Experiments'}, 'FontSize', plotSettings.yLabelFontSize-3);
set(gca, 'FontSize', plotSettings.gcaFontSize-3);
ylim([-1 1]);

end