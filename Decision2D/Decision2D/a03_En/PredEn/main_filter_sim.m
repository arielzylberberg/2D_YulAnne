%% For S1, use dif_rel_incl_find=1:2
init_path;
Bank0 = FilterBankRegrSim;
Bank0.batch_apply_filter( ...
    'subj', 'S1', ...
    'dif_rel_incl_find', 1:2, ...
    'to_use_roni', false);


%% For S2, use dif_rel_incl_find=1
init_path;
Bank0 = FilterBankRegrSim;
Bank0.batch_apply_filter( ...
    'subj', 'S2', ...
    'dif_rel_incl_find', 1, ...
    'to_use_roni', false);

%% For S3, use dif_rel_incl_find=1
init_path;
Bank0 = FilterBankRegrSim;
Bank0.batch_apply_filter( ...
    'subj', 'S3', ...
    'dif_rel_incl_find', 1, ...
    'to_use_roni', false);

%% rsync commands
% rsync -avz -e ssh "../Data_2D/sTr/dot5s/" "yul@pat.shadlen.zi.columbia.edu:~/"
% rsync -avz -e ssh "../Data_2D/Fit.Filter.Motion.FilterBankRegr/" "yul@pat.shadlen.zi.columbia.edu:~/s/Decision2D/Data_2D/Fit.Filter.Motion.FilterBankRegr/"
% rsync -avz -e ssh "yul@pat.shadlen.zi.columbia.edu:~/s/Decision2D/Data_2D/Fit.Filter.Motion.FilterBankRegr/" "../Data_2D/Fit.Filter.Motion.FilterBankRegr/"
% rsync -avz -e ssh "yul@pat.shadlen.zi.columbia.edu:~/s/Decision2D/Data_2D/FilterBankRegrSim/" "../Data_2D/FilterBankRegrSim/"
