function run_all_analysis()
% function run_all_analysis()
% 07-2020 Ariel Zylberberg wrote it (ariel.zylberberg@gmail.com)

if 0
    % run_prep_data(); % preps the data
    run_fit_2D(); % run this in cluster with runcode.sh; then manually move files to folder 'from_fits'
end

overwrite = 1;
run_eval_best(overwrite); % evals best and saves more outputs
run_calc_fine(overwrite); % calculates finer model predictions
redo_calc = 1;
run_calc_like_not_pred(redo_calc); % calcs and plots likelihood comparison

% figures for paper
run_fig2(1,0,0);
run_fig2(2,0,0);
run_fig2(2,1,1);
run_fig2(2,1,0);

run_fig2_per_suj(); 

end