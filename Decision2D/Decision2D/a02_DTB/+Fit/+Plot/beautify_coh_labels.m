function labels = beautify_coh_labels(label_values)
% labels = beautify_coh_labels(label_values)

    is_motion = any(ismember( ...
        label_values, ...
        [0.032, 0.064, 0.128, 0.256, 0.512]));
    
    if is_motion
        labels = csprintf('%1.3g', label_values * 100);
    else
        labels = csprintf('%1.3g', label_values);
    end
    labels(strcmp(labels, '0.0')) = {'0'};
    labels(strcmp(labels, 'NaN')) = {''};
end