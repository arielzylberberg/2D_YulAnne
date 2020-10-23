function main()

clear;
init_path;

%% Convert data
main_combine_data_RT_VD;

%% -- 2D RT - Ser vs Par, model-free Figure and Table
% Real & simulated data

% README: 
% Fits Fig X - takes days to run
Fit.D2.RT.Td2Tnd.main_td2tnd;

% Loads fitted parameters from above and produces predictions for Figs X
Fit.D2.RT.Td2Tnd.main_export_td2tnd_pred_data;

%% Misc
% main_check_Scr_info;
% main_n_days;

