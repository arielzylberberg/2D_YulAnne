% main_en

%% Derive en
Fit.Filter.Motion.main_filter;

%% Derive en - nonnegative logistic - current method of choice
Fit.Filter.Motion.main_blogistic_btw_tasks;

% %% Derive en - octant
% IxnKernel.get_ME_octant;

%% Check en: plot ch by mean en
Fit.Filter.Motion.main_plot_ch_by_en_data;

%% Simple en & revcor simulation
main_pred_en_simple;

%% == Simulate full 5s ME & CE
%% Make full 5s stimuli
main_simulate_xyct;

%% Run filters on the simulated stimuli
main_filter_sim;



%% == Dilution analysis - compare choice slope
% %% Just compute the logistic interaction term for each duration
% IxnKernel.PAcqD2.main_ixn_by_dur; % 
% 
% %% Thresholding RT dif by percentile & comparing slope
% s = GLM.Short.SlopeThresRT;
% s.main_slope_aft_thres_rt_dif;
% 
% %% RT Dilution analysis - compare between conditions
% IxnKernel.PAcqD2.main_p_acq_d2_by_cond;

%% Simple RevCor
main_revcor_simple;
main_plot_lev_t;

%% == IxnKernel
%% Fit Dtb with en for IxnKernel
% Fit.main_scale_en; % Run just once - already hard coded in Data.Consts

%%
Fit.main_dtb_en_1D;
% 
% %% Simple GLM
% EnIxn.main_EnIxn;
% 
% %% Comapre leverage between 1D and 2D - not working
% EnIxn.main_compare_lev_segment;
% 
% %% Derive p_use from 1D BoundedEn fits
% IxnKernel.main_ixn_kernel;

%% Cum En
main_cum_en_logit;

%% XCorr
main_xcorr;

%% XCorr by motion and color lags
main_xcorr_by_lags;

%% Predict RevCorrSimple and XCorr based on Fit.D1.BoundedEn
main_pred_en;