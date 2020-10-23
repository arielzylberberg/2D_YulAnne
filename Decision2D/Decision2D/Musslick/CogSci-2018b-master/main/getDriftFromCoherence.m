function drift = getDriftFromCoherence(coherence, task) 
% a function to that allows to convert coherence into drift

    if(strcmp(task, 'color'))
        drift = coherence;
        %% 

    elseif(strcmp(task, 'motion'))
        drift = coherence;
    end

end