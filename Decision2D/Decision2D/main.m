function main()

clear;
init_path;

%% Remove outliers
Fit.Check.main_find_outliers;

%%
main_combine_data; % Combine data into one file

%% == GLM
%% -- 1D, 2D choice GLM Table
GLM.Ch.main_ch;
GLM.Ch.main_heatmap;

%% -- 1D, 2D reaction time GLM Table
GLM.RT.main_rt;

% average difference between ndim & between difficulty levels within sh
Fit.main_compare_RT_sh;

%% == 1D DTB - RT
Fit.main_dtb_1D_RT; % TODO

%% == 1D DTB - Short
Fit.main_dtb_1D_sh;

%% == 2D Ser vs Par


%% -- 2D RT - Ser vs Par, model-free Figure and Table
% Real & simulateed data
Fit.D2.RT.Td2Tnd.main_td2tnd;

% Figure (condition inclusion)
Fit.D2.RT.Td2Tnd.fig_cond_train_vs_valid;

%% Plot all sh (pooled)
Fit.CompareModels.main_plot_sh_all_subj;

%% -- 2D IrrSep - Fit on each irr condition separately
Fit.main_dtb_irrsep;

%% -- 2D RT - fit models
%% Fit Ser/Par Dtb in RT, make prediction, and compare
Fit.main_dtb_RT;

%% -- Inh - Sanity check
Fit.D2.Inh.main_inh_sanitycheck;

%% -- Inh - Time slicing model
Fit.main_dtb_inh_timeslice;

%% Compare models - cost bar plot (all) & scatterplots (inh)
Fit.main_compare_dtb_all;
Fit.main_compare_dtb_inh;
Fit.main_save_pred_data;
Fit.CompareModels.main_plot_pooled_ser_vs_par; % RT
Fit.CompareModels.main_plot_best_dtb;

%% RT-vs-RT plots across models
% Fit.main_compare_dtb_plots;
GLM.ModelFreeChRt.get_pred_pdf_all_inh; % Run only once
GLM.ModelFreeChRt.main_modelfree_rt;
GLM.ModelFreeChRt.main_modelfree_ch;
GLM.ModelFreeChRt.main_combine_fig_ch_rt;
GLM.ModelFreeChRt.main_combine_indiv_fig_ch_rt;
GLM.ModelFreeChRt.main_rt_by_rt_slope; % stats on model 2 regression slope
TimeDepAccu.main_tda;

%% Schematic
ConceptFig.main_schematic;
ConceptFig.main_pred_cartoon;

%% Reward rate
Fit.main_save_pred_from_th_model
Fit.RewardRate.main_reward_rate;

%% == Short Expr RT - Buffer length
Fit.main_dtb_short;

%% == Short Expr - Time-Dependent Accuracy
TimeDepAccu.main_tda;

% Compare ECDF by accuracy on each dim, btw pred & data
TimeDepAccu.main_ecdf_accu;

%% == Acq - Simple energy analyses
main_en;

%% == Eye
main_eye;


%% == 2D RT - Ser model & figure

%% == 2D RT - Model comparison figure (bar plot for BIC)

%% 