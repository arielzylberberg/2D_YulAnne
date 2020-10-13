%% Suppl Fig 8 - Model-free comparison of uni- vs. bimanual task
clear
close all
clc

load('../data/RT_task/data_RT.mat');

IDs = [3 4 5 6 7 9 16 18];

trials = dataset == 2; % select hand data only


% get absolute coherence levels
uMotCoh = uniquetol(abs(coh_motion(trials))); % unsigned motion coherence levels
uColCoh = uniquetol(abs(coh_color(trials))); % unsigned color coherence levels
sMotCoh = unique(coh_motion(trials));
sColCoh = unique(coh_color(trials));

% group into weak vs. strong coherence
Motion_grouped = [uMotCoh(1:3) uMotCoh(4:6)];
Color_grouped = [uColCoh(1:3) uColCoh(4:6)];
    

% Only include correct trials (or 0% coherence) in RT analyses
correct = (corr_motion | coh_motion == 0) & (corr_color | coh_color == 0);


%% get results for each individual participant
for i = 1:length(IDs)
    
    subjID = IDs(i);
    
    %% get results from uni-/bimanual task
    for k = 1:2
        if k == 1 % unimanual
            trialIdx = trials & group == IDs(i) & bimanual == 0 & ~isnan(RT); % ignore miss trials
        elseif k == 2 % bimanual
            trialIdx = trials & group == IDs(i) & bimanual == 1 & ~isnan(RT) & RT1 ~= RT; % ignore miss trials
        end

        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %--------------------- Choice Performance ------------------------%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % MOTION: % "right" choices       
        % fit psychometric curve for each color coherence level
        for c = 1:length(uColCoh)
            trialIdx_choice = trialIdx & ismember(abs(coh_color), uColCoh(c));
            MotionChoice(c,:,i,k) = glmfit(coh_motion(trialIdx_choice),[choice_motion(trialIdx_choice) == 1 ones(sum(trialIdx_choice),1)],'binomial','logit');            
        end
        
        % COLOR: % "blue" choices
        % fit psychometric curve for each motion coherence level
        for m = 1:length(uMotCoh)
            trialIdx_choice = trialIdx & ismember(abs(coh_motion), uMotCoh(m));
            ColorChoice(m,:,i,k) = glmfit(coh_color(trialIdx_choice),[choice_color(trialIdx_choice) == 1 ones(sum(trialIdx_choice),1)],'binomial','logit');            
        end
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %----------------------------- RTs -------------------------------%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % MOTION: RTs for each motion x color coherence
        for m = 1:length(uMotCoh)
            for c = 1:size(Color_grouped,2)
                trialIdx_rt = trialIdx & correct & abs(coh_motion) == uMotCoh(m) & ismember(abs(coh_color),Color_grouped(:,c));
                RTmotion(i,m,c,k) = mean(RT(trialIdx_rt));
            end
        end
        
        % COLOR: RTs for each motion x color coherence
        for c = 1:length(uColCoh)
            for m = 1:size(Motion_grouped,2)
                trialIdx_rt = trialIdx & correct & abs(coh_color) == uColCoh(c) & ismember(abs(coh_motion),Motion_grouped(:,m));
                RTcolor(i,c,m,k) = mean(RT(trialIdx_rt));
            end
        end
        
    end
end

%% set up figure properties
set(0,'DefaultAxesBox', 'off',...
    'DefaultAxesFontSize',16,...
    'DefaultFigureUnits', 'normalized', ...
    'DefaultFigurePosition', [0.1, 0.1, 0.65, 0.95]);

colors = [[74 101 143]; % blue
    [144 66 41]]/255; % red

colors_uni = ...
    [[163 176 198];
    [74 101 143]]/255; % dark blue

colors_bi = ...
    [[211 179 170];
    [144 66 41]]/255; % dark red

MarkerSize = 10;


%% Plot results
% choice sensitivity as function of current x other dimension
figure(1);
subplot(2,2,1); axis square; 

hold all
h1 = errorbar(1:length(uMotCoh), mean(ColorChoice(:,2,:,1),3),std(ColorChoice(:,2,:,1),[],3)/sqrt(size(ColorChoice,3)),'o','Color',colors(1,:),'LineWidth',2,'MarkerFaceColor',colors(1,:), 'MarkerSize', MarkerSize); % unimanual
h2 = errorbar(1:length(uMotCoh), mean(ColorChoice(:,2,:,2),3),std(ColorChoice(:,2,:,2),[],3)/sqrt(size(ColorChoice,3)),'o','Color',colors(2,:),'LineWidth',2,'MarkerFaceColor',colors(2,:), 'MarkerSize', MarkerSize); % bimanual
% placeholder plots for legends
h3 = plot(-5,-5,'o','Color',colors(1,:), 'MarkerFaceColor',colors(1,:),'LineWidth',2, 'MarkerSize', MarkerSize);
h4 = plot(-5,-5,'o','Color',colors(2,:), 'MarkerFaceColor',colors(2,:),'LineWidth',2, 'MarkerSize', MarkerSize);
set(gca,'Xlim', [-.05 length(uMotCoh)+1], 'XTick', 1:length(uMotCoh), 'XTickLabels', {},'Ylim', [7 17],'TickDir','out');
ylab = ylabel('Color sensitivity'); 
leg = legend([h3 h4], 'Unimanual','Bimanual', 'Location', 'SouthWest', 'box','off');
title(leg,'Task version', 'FontSize',16);

subplot(2,2,2); axis square; hold all
h1 = errorbar(1:length(uColCoh), mean(MotionChoice(:,2,:,1),3),std(MotionChoice(:,2,:,1),[],3)/sqrt(size(MotionChoice,3)),'o','Color',colors(1,:),'LineWidth',2, 'MarkerFaceColor',colors(1,:), 'MarkerSize', MarkerSize); % unimanual
h2 = errorbar(1:length(uColCoh), mean(MotionChoice(:,2,:,2),3),std(MotionChoice(:,2,:,2),[],3)/sqrt(size(MotionChoice,3)),'o','Color',colors(2,:),'LineWidth',2, 'MarkerFaceColor',colors(2,:), 'MarkerSize', MarkerSize); % bimanual
set(gca,'Xlim', [-.05 length(uColCoh)+1], 'XTick', 1:length(uColCoh), 'XTickLabels', {},'Ylim', [12.5 37.5],'TickDir','out');
ylab = ylabel('Motion sensitivity'); 


% RTs for current & other dimension
subplot(2,2,3); axis square; hold all
for c = 1:size(Color_grouped,2)
    errorbar(1:length(uMotCoh), mean(RTmotion(:,:,c,1),1),std(RTmotion(:,:,c,1),[],1)/sqrt(size(RTmotion,1)),'o','Color',colors_uni(c,:),'LineWidth',2, 'MarkerFaceColor',colors_uni(c,:), 'MarkerSize', MarkerSize); % unimanual
    errorbar(1:length(uMotCoh), mean(RTmotion(:,:,c,2),1),std(RTmotion(:,:,c,2),[],1)/sqrt(size(RTmotion,1)),'o','Color',colors_bi(c,:),'LineWidth',2, 'MarkerFaceColor',colors_bi(c,:), 'MarkerSize', MarkerSize); % bimanual
end
set(gca,'Xlim', [-.05 length(uMotCoh)+1], 'XTick', 1:length(uMotCoh), 'XTickLabels', round(100*uMotCoh)/100,'XTickLabelRotation',45, 'Ylim', [.4 2.6], 'YTick', .5:.5:2.5,'TickDir','out');
xlab = xlabel('Motion strength (|coh|)');
ylab = ylabel('RT (sec)'); 
% placeholder plots for legends
h5 = plot(-5,-5,'o','Color',[1 1 1]/1.3, 'MarkerFaceColor',[1 1 1]/1.3,'LineWidth',2, 'MarkerSize', MarkerSize);
h6 = plot(-5,-5,'o','Color',[1 1 1]/2.5, 'MarkerFaceColor',[1 1 1]/2.5,'LineWidth',2, 'MarkerSize', MarkerSize);

leg = legend([h5 h6], 'Weak','Strong', 'Location', 'SouthWest','box','off');
title(leg,'             Other dimension','FontSize',16);
legpos = get(leg, 'Position');
set(leg, 'Position', [legpos(1)-.04 legpos(2:4)]);
subplot(2,2,4); axis square; hold all
for m = 1:size(Motion_grouped,2)
    h1 = errorbar(1:length(uColCoh), mean(RTcolor(:,:,m,1),1),std(RTcolor(:,:,m,1),[],1)/sqrt(size(RTcolor,1)),'o','Color',colors_uni(m,:),'LineWidth',2, 'MarkerFaceColor',colors_uni(m,:), 'MarkerSize', MarkerSize); % unimanual
    h2 = errorbar(1:length(uColCoh), mean(RTcolor(:,:,m,2),1),std(RTcolor(:,:,m,2),[],1)/sqrt(size(RTcolor,1)),'o','Color',colors_bi(m,:),'LineWidth',2, 'MarkerFaceColor',colors_bi(m,:), 'MarkerSize', MarkerSize); % bimanual
end

set(gca,'Xlim', [-.05 length(uColCoh)+1], 'XTick', 1:length(uColCoh), 'XTickLabels', round(100*uColCoh)/100,'XTickLabelRotation',45, 'Ylim', [.4 2.6], 'YTick', .5:.5:2.5,'TickDir','out');
xlab = xlabel('Color strength (|coh|)'); 
ylab = ylabel('RT (sec)'); 




