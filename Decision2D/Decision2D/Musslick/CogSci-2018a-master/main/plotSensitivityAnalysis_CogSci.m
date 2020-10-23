function plotSensitivityAnalysis(offset, distanceMeasure, goodnessOfFit, c_diff, measureOfFit, offsetTicks, distanceMeasureTicks, offsetLabel, distanceMeasureLabel, c1, c2, assumedParameterization, titleLabel)
    
    % load plot settings;
    plotsettings;
    
    % open figure
    fig = figure(1);
    set(fig, 'Position', [100 100 260 200]);

    % get rid of nans
%     nanVal = 10^16;
%     goodnessOfFit(isnan(goodnessOfFit)) = nanVal;
    
    % plot data
    h = imagesc(goodnessOfFit); hold on;
    minFit = -5;
    maxFit = 7;
    caxis([minFit maxFit]);
    
    % identify flipped instances
    correctDirectionX = [];
    correctDirectionY = [];
    falseDirectionX = [];
    falseDirectionY = [];
    
    for i = 1:length(c_diff(:))
        
        if(c_diff(i) > 0)
            [correctDirectionX(end+1), correctDirectionY(end+1)] = ind2sub(size(c_diff),i);
        else
            [falseDirectionX(end+1), falseDirectionY(end+1)] = ind2sub(size(c_diff),i);
        end
        
    end
    
%     plot(correctDirectionY, correctDirectionX, 'o','Color', 'k', 'MarkerSize', 5);
%     plot(falseDirectionY, falseDirectionX, 'x','Color', 'k', 'MarkerSize', 5);

    % find indices for which cost relationship is flipped
    switch measureOfFit
        case 'MSE'
            measureOfFitLabel = 'Mean Squared Error';
        case 'delta_c_hat'
            measureOfFitLabel = '(c2-c1)';
            
%             goodnessOfFit_range = max(goodnessOfFit(:)) - min(goodnessOfFit(:));
%             distanceToFlip = 0 - min(goodnessOfFit(:));
%             distancetoCorrect = (c2-c1) - min(goodnessOfFit(:));
%             sweetSpot = distanceToFlip / goodnessOfFit_range;
%             trueSpot = distancetoCorrect / goodnessOfFit_range;
            
            goodnessOfFit_range = maxFit - minFit;
            distanceToFlip = 0 - minFit;
            distancetoCorrect = (c2-c1) - minFit;
            sweetSpot = distanceToFlip / goodnessOfFit_range;
            trueSpot = distancetoCorrect / goodnessOfFit_range;

        otherwise
          warning('Could not identify critical cases');  
    end

    % set ticks
    set(gca, 'XTick', offset) 
    set(gca, 'YTick', distanceMeasure)
    set(gca,  'XTickLabel', offsetTicks);
    set(gca, 'YTickLabel', distanceMeasureTicks);
    
    % plot assumed parameterizaiton
    if(exist('assumedParameterization', 'var'))
        
        if(~isempty(assumedParameterization))
        
            [nRows nCols] = size(goodnessOfFit);
            plot([assumedParameterization(1) assumedParameterization(1)], [0 nRows], '--k', 'LineWidth', 2);
            plot([0 nCols], [assumedParameterization(2) assumedParameterization(2)], '--k', 'LineWidth', 2);
            plot([assumedParameterization(1) assumedParameterization(1)], [assumedParameterization(2) assumedParameterization(2)], 'o', 'MarkerSize', 10, 'Color', 'k', 'LineWidth', 2);

        end
        
    end
    
    % plot diagonal
    if(exist('assumedParameterization', 'var'))
        if(~isempty(assumedParameterization))
            padding = 10;
            plot([min([offset distanceMeasure])-padding max([offset distanceMeasure])+padding], [min([offset distanceMeasure])-padding max([offset distanceMeasure])+padding], '-k', 'LineWidth', 2);
            [min([offset distanceMeasure]) min([offset distanceMeasure])]
            [max([offset distanceMeasure]) max([offset distanceMeasure])]
        end
    end
    
    % set colorbar
    cMap = cool;
    cMap = cMap(fliplr(1:size(cMap,1)), :);
    res = 1000;
    cMap_long = nan(res, 3);
    cMap_long(:,1) = linspace(cMap(1,1), cMap(end,1), res);
    cMap_long(:,2) = linspace(cMap(1,2), cMap(end,2), res);
    cMap_long(:,3) = linspace(cMap(1,3), cMap(end,3), res);
    cMap = cMap_long;
    
    if(exist('sweetSpot', 'var') && exist('trueSpot', 'var'))
        nColors = size(cMap, 1);
        coolMap = cool;
        goodMap = winter;
        badMap = autumn;
        
        goodToTrueSpot = (round(sweetSpot*trueSpot) + 1) : nColors;
        trueSpotToSweetSpot = (round(sweetSpot*nColors) + 1) :  round(trueSpot*nColors);
        sweetSpotToBad = 1:(round(sweetSpot*nColors));
        
        if(~any(trueSpotToSweetSpot <= 0) && ~any(goodToTrueSpot <= 0) && ~any(sweetSpotToBad <= 0) && ...
            ~any(isnan(trueSpotToSweetSpot)) && ~any(isnan(goodToTrueSpot)) && ~any(isnan(sweetSpotToBad)))

            for col = 1:3
                cMap(goodToTrueSpot, col) = linspace(goodMap(1, col),  coolMap(1, col), length(goodToTrueSpot));
                cMap(trueSpotToSweetSpot, col) = linspace(goodMap(end, col),  goodMap(1, col), length(trueSpotToSweetSpot));
                cMap(sweetSpotToBad, col) = linspace(badMap(1, col),  badMap(end, col), length(sweetSpotToBad));
            end
            
        end
    end
    
    % mark nans
    
    colormap(cMap);

    hcb = colorbar;
    set(h,'AlphaData',~isnan(goodnessOfFit))
    
    % set labels
    if(exist('titleLabel', 'var'))
        title(titleLabel, 'FontSize', plotSettings.titleFontSize);
    else
        title(['$c_1 =' num2str(c1) ', c_2 = ' num2str(c2) '$'], 'interpreter', 'latex', 'FontSize', plotSettings.titleFontSize);
    end
    xlabel(offsetLabel, 'FontSize', plotSettings.xLabelFontSize-1);
    ylabel(distanceMeasureLabel, 'FontSize', plotSettings.yLabelFontSize-1);
%     ylabel(hcb, measureOfFitLabel, 'FontSize', plotSettings.yLabelFontSize)

    hold off;
    
end