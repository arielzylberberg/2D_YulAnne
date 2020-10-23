function [params, sequenceName]= generateToySequence(numTrialsPerMiniBlock, numMiniBlocks, frequency, congruentTrials, varargin)

    % compute trial number
    nSwitchTrials = round(numMiniBlocks * frequency);
    nRepetitionTrials = numMiniBlocks - nSwitchTrials;
    
    trialCounter = 1;
    
    % determine task transitions
    transitions = [ones(1, nSwitchTrials) zeros(1, nRepetitionTrials)];
    
    order = randperm(length(transitions));
    
    % determine congruency
    congruency = repmat([1 0], 1, numMiniBlocks/2);
    
    transition = [];
    tasks = [];
    color = [];
    motion = [];
    correct = [];
    incongruent = [];
    miniBlockTrials = [];
    miniBlockSize = [];
    miniBlocks = [];
    trials = [];
    
    % loop through each miniBlock
    trial = 1;
    for miniBlock = 1:numMiniBlocks
        
        % loop through each trial per miniBlock
        trialsPerMiniBlock = (numTrialsPerMiniBlock(mod(miniBlock, length(numTrialsPerMiniBlock))+1));
        for miniBlockTrial = 1:trialsPerMiniBlock
            
            % determine task
            if(miniBlockTrial == 1)
                if(miniBlock == 1)
                        tasks = 1;      % begin with task 1
                else
                    if(transitions(miniBlock) == 1)
                        tasks = [tasks mod(tasks(end), 2)+1];
                    else
                        tasks = [tasks tasks(end)];
                    end
                end
            else
                tasks = [tasks tasks(end)];
            end
            
            % set transition condtition
            transition = [transition transitions(miniBlock)];
            
            % determine incongruency
            if(congruentTrials)
                if(miniBlockTrial == 1)
                    incongruentSample = congruency(miniBlock);
                else
                    incongruentSample = mod(incongruent(end)+1, 2);
                end
                incongruent = [incongruent incongruentSample];
            else
                incongruent = [incongruent 1];
            end
            
            % determine direction 
            if(incongruent(end) == 1)
                color = [color 1];
                motion = [motion 2];
            else
                color = [color 1];
                motion = [motion 1];    
            end
            
            % determine correct response
            if(tasks(end) == 1)
                correct = [correct color(end)];
            else
                correct = [correct motion(end)];
            end
            
            miniBlockTrials = [miniBlockTrials miniBlockTrial];
            miniBlocks = [miniBlocks miniBlock];
            miniBlockSize = [miniBlockSize trialsPerMiniBlock];
            trials = [trials trial];
            trial = trial + 1;
        end
        
    end
    
    params.CTS.Sequence{2}.TaskTransition = transition;
    params.CTS.Sequence{2}.Task = tasks;
    params.CTS.Sequence{2}.curColor = color;
    params.CTS.Sequence{2}.curMotion= motion;
    params.CTS.Sequence{2}.correctResponse= correct;
    params.CTS.Sequence{2}.miniBlockTrial = miniBlockTrials;
    params.CTS.Sequence{2}.miniBlock = miniBlocks;
    params.CTS.Sequence{2}.miniBlockSize= miniBlockSize;
    params.CTS.Sequence{2}.trials= trials;
    params.CTS.Sequence{2}.incongruency= incongruent;
    params.CTS.Sequence{2}.frequency = frequency;
    
    %% plot
    if(0)
        imagesc([params.CTS.Sequence{2}.Task; ...
                       params.CTS.Sequence{2}.TaskTransition; ...
                       params.CTS.Sequence{2}.curColor; ...
                       params.CTS.Sequence{2}.curMotion; ...
                       params.CTS.Sequence{2}.correctResponse; ...
                       params.CTS.Sequence{2}.incongruency]);
        colorbar;
    end
    
    sequenceName = ['logfiles/toy_sequence_' num2str(frequency) '.mat'];
    save(sequenceName, 'params');

end