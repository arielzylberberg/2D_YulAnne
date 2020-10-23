function [overallRT_global, overallER_global, incongruencyCostRT_global, incongruencyCostER_global, switchCostRT_global, switchCostER_global, ...
               overallRT_local, overallER_local, incongruencyCostRT_local, incongruencyCostER_local, switchCostRT_local, switchCostER_local, ...
               relActivation_global, relActivation_local, irrelActivation_global, irrelActivation_local,  ...
               allDrift, transition, incongruency, allActivation] = CogSci_Simulation2_analysisLocal(gain, ddmpParams, sequenceFile, tau, inputWeight)

    [optimizationCriterion, performance, allRR, allmeanRT, allmeanAccuracy, allActivation, allDrift] = gainInvestigation_simulateSequence(gain, ddmpParams, sequenceFile, tau, inputWeight);
    load(sequenceFile);
    
    transition = params.CTS.Sequence{2}.TaskTransition;
    incongruency = params.CTS.Sequence{2}.incongruency;
    miniBlockTrial = params.CTS.Sequence{2}.miniBlockTrial;
    task = params.CTS.Sequence{2}.Task;
    
    if(isfield(params.CTS.Sequence{2}, 'local'))
        localSequence = 1;
    else
        localSequence = 0;
    end
    
    
    % grab relevant activation
    relActivation = nan(1, size(allActivation,2));
    irrelActivation = nan(1, size(allActivation,2));
    for t = 1:size(allActivation,2)
        relActivation(t) = allActivation(task(t),t);
        irrelTask = mod(task(t), 2)+1;
        irrelActivation(t) = allActivation(irrelTask,t);
    end
    
    %% GLOBAL EFFECTS
    
    % make sure to account for potential sequence biases (full-crossing between factors)
    
    repetition_congruent_trials_global = find(transition == 0 & miniBlockTrial == 1 & incongruency == 0);
    repetition_incongruent_trials_global = find(transition == 0 & miniBlockTrial == 1 & incongruency == 1);
    switch_congruent_trials_global = find(transition == 1 & miniBlockTrial == 1 & incongruency == 0);
    switch_incongruent_trials_global = find(transition == 1 & miniBlockTrial == 1 & incongruency == 1);
    
%     congruent_repetition_trials_global = find(transition == 0 & incongruency == 0);
%     congruent_switch_trials_global = find(transition == 1 & incongruency == 0);
%     incongruent_repetition_trials_global = find(transition == 0 & incongruency == 1);
%     incongruent_switch_trials_global = find(transition == 1 & incongruency == 1);
    
    min_repetition_trials_global = min([length(repetition_congruent_trials_global) length(repetition_incongruent_trials_global)]);
    min_switch_trials_global = min([length(switch_congruent_trials_global) length(switch_incongruent_trials_global)]);
%     min_congruent_trials_global = min([length(congruent_repetition_trials_global) length(congruent_switch_trials_global)]);
%     min_incongruent_trials_global = min([length(incongruent_repetition_trials_global) length(incongruent_switch_trials_global)]);

    repetition_trials_global = [repetition_congruent_trials_global(randperm(min_repetition_trials_global)) repetition_incongruent_trials_global(randperm(min_repetition_trials_global))];
    switch_trials_global = [switch_congruent_trials_global(randperm(min_switch_trials_global)) switch_incongruent_trials_global(randperm(min_switch_trials_global))];
%     congruent_trials_global = find(incongruency == 0 & transition == 0);
%     incongruent_trials_global = find(incongruency == 1 & transition == 0);
    congruent_trials_global = find(incongruency == 0);
    incongruent_trials_global = find(incongruency == 1);    
    overall_trials_global = find(transition == 0 | transition == 1);
    
    % compute means for each condition
    
    repetitionRT_mean_global = mean(allmeanRT(repetition_trials_global));
    switchRT_mean_global = mean(allmeanRT(switch_trials_global));
    repetitionAccuracy_mean_global = mean(allmeanAccuracy(repetition_trials_global));
    switchAccuracy_mean_global = mean(allmeanAccuracy(switch_trials_global));
    repetitionDrift_mean_global = mean(abs(allDrift(repetition_trials_global)));
    switchDrift_mean_global = mean(abs(allDrift(switch_trials_global)));

    congruentRT_mean_global = mean(allmeanRT(congruent_trials_global));
    incongruentRT_mean_global = mean(allmeanRT(incongruent_trials_global));
    congruentAccuracy_mean_global = mean(allmeanAccuracy(congruent_trials_global));
    incongruentAccuracy_mean_global = mean(allmeanAccuracy(incongruent_trials_global));
    congruentDrift_mean_global = mean(abs(allDrift(congruent_trials_global)));
    incongruentDrift_mean_global = mean(abs(allDrift(incongruent_trials_global)));
    
    switchER_mean_global = 1 - switchAccuracy_mean_global;
    repetitionER_mean_global = 1 - repetitionAccuracy_mean_global;
    congruentER_mean_global = 1 - congruentAccuracy_mean_global;
    incongruentER_mean_global = 1 - incongruentAccuracy_mean_global;
    
    switchCostRT_global = switchRT_mean_global - repetitionRT_mean_global;
    incongruencyCostRT_global = incongruentRT_mean_global - congruentRT_mean_global;
    overallRT_global = mean(allmeanRT(overall_trials_global));
    
    switchCostER_global = switchER_mean_global - repetitionER_mean_global;
    incongruencyCostER_global = incongruentER_mean_global - congruentER_mean_global;
    overallER_global = 1-mean(allmeanAccuracy(overall_trials_global));
    
    relActivation_global = mean(relActivation);
    irrelActivation_global = mean(irrelActivation);
    
    
    worstER_global = mean(1-allmeanAccuracy(transition == 1 & miniBlockTrial == 1 & incongruency == 1));
    worstRT_global = mean(allmeanRT(transition == 1 & miniBlockTrial == 1 & incongruency == 1));
    disp(['error rate on inc switch trials:' num2str(worstER_global)]);
    disp(['reaction time on inc switch trials:' num2str(worstRT_global)]);
%     repetitionDrift_mean
%     switchDrift_mean
%     repetitionDrift_mean-switchDrift_mean

    %% LOCAL
     
    if(localSequence)
        local = params.CTS.Sequence{2}.local;
    
        relActivation_local = mean(relActivation(local == 1));
        irrelActivation_local = mean(irrelActivation(local == 1));
        
        % make sure to account for potential sequence biases (full-crossing between factors)

        repetition_congruent_trials_local = find(transition == 0 & miniBlockTrial == 1 & incongruency == 0 & local == 1);
        repetition_incongruent_trials_local = find(transition == 0 & miniBlockTrial == 1 & incongruency == 1 & local == 1);
        switch_congruent_trials_local = find(transition == 1 & miniBlockTrial == 1 & incongruency == 0 & local == 1);
        switch_incongruent_trials_local = find(transition == 1 & miniBlockTrial == 1 & incongruency == 1 & local == 1);
    %     
    %     congruent_repetition_trials_local = find(transition == 0 & incongruency == 0 & local == 1);
    %     congruent_switch_trials_local = find(transition == 1 & incongruency == 0 & local == 1);
    %     incongruent_repetition_trials_local = find(transition == 0 & incongruency == 1 & local == 1);
    %     incongruent_switch_trials_local = find(transition == 1 & incongruency == 1 & local == 1);

        min_repetition_trials_local = min([length(repetition_congruent_trials_local) length(repetition_incongruent_trials_local)]);
        min_switch_trials_local = min([length(switch_congruent_trials_local) length(switch_incongruent_trials_local)]);
    %     min_congruent_trials_local = min([length(congruent_repetition_trials_local) length(congruent_switch_trials_local)]);
    %     min_incongruent_trials_local = min([length(incongruent_repetition_trials_local) length(incongruent_switch_trials_local)]);

        repetition_trials_local = [repetition_congruent_trials_local(randperm(min_repetition_trials_local)) repetition_incongruent_trials_local(randperm(min_repetition_trials_local))];
        switch_trials_local = [switch_congruent_trials_local(randperm(min_switch_trials_local)) switch_incongruent_trials_local(randperm(min_switch_trials_local))];
        congruent_trials_local = find(incongruency == 0 & local == 1);
        incongruent_trials_local = find(incongruency == 1 & local == 1);
        overall_trials_local = find(transition == 0 & local == 1);

        % compute means for each condition

        repetitionRT_mean_local = mean(allmeanRT(repetition_trials_local));
        switchRT_mean_local = mean(allmeanRT(switch_trials_local));
        repetitionAccuracy_mean_local = mean(allmeanAccuracy(repetition_trials_local));
        switchAccuracy_mean_local = mean(allmeanAccuracy(switch_trials_local));
        repetitionDrift_mean_local = mean(abs(allDrift(repetition_trials_local)));
        switchDrift_mean_local = mean(abs(allDrift(switch_trials_local)));

        congruentRT_mean_local = mean(allmeanRT(congruent_trials_local));
        incongruentRT_mean_local = mean(allmeanRT(incongruent_trials_local));
        congruentAccuracy_mean_local = mean(allmeanAccuracy(congruent_trials_local));
        incongruentAccuracy_mean_local = mean(allmeanAccuracy(incongruent_trials_local));
        congruentDrift_mean_local = mean(abs(allDrift(congruent_trials_local)));
        incongruentDrift_mean_local = mean(abs(allDrift(incongruent_trials_local)));

        switchER_mean_local = 1 - switchAccuracy_mean_local;
        repetitionER_mean_local = 1 - repetitionAccuracy_mean_local;
        congruentER_mean_local = 1 - congruentAccuracy_mean_local;
        incongruentER_mean_local = 1 - incongruentAccuracy_mean_local;

        switchCostRT_local = switchRT_mean_local - repetitionRT_mean_local;
        incongruencyCostRT_local = incongruentRT_mean_local - congruentRT_mean_local;
        overallRT_local = mean(allmeanRT(overall_trials_local));

        switchCostER_local = switchER_mean_local - repetitionER_mean_local;
        incongruencyCostER_local = incongruentER_mean_local - congruentER_mean_local;
        overallER_local = 1-mean(allmeanAccuracy(overall_trials_local));

    else
        
        relActivation_local = nan;
        irrelActivation_local = nan;

        repetitionRT_mean_local = nan;
        switchRT_mean_local = nan;
        repetitionAccuracy_mean_local = nan;
        switchAccuracy_mean_local = nan;
        repetitionDrift_mean_local = nan;
        switchDrift_mean_local = nan;

        congruentRT_mean_local = nan;
        incongruentRT_mean_local = nan;
        congruentAccuracy_mean_local = nan;
        incongruentAccuracy_mean_local = nan;
        congruentDrift_mean_local = nan;
        incongruentDrift_mean_local = nan;

        switchER_mean_local = nan;
        repetitionER_mean_local = nan;
        congruentER_mean_local = nan;
        incongruentER_mean_local = nan;

        switchCostRT_local = nan;
        incongruencyCostRT_local = nan;
        overallRT_local = nan;

        switchCostER_local = nan;
        incongruencyCostER_local = nan;
        overallER_local = nan;
    end
end