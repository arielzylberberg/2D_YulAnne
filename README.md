# 2D Decision

## Figure 2 and fits for Figure 2 supplement 3
 * All MATLAB scripts are in folder `model_fit_RT_models`: `run_all_analysis.m` will execute all code that is required to plot the serial and parallel choice-RT fits 
   * Note: By default, the `run_fit_2D()` function will not be called since fitting takes several days to complete. To re-fit the data, run `run_fit_2D.m` on a cluster (see runcode.sh), then manually move files with new fits to `from_fits` folder 


## Figure 2 supplements 1-5
* In MATLAB, run `main_fig2supp1_5.m` to load and use fitted parameters to export predictions that will be used for plotting.
* Then in Python, set `Decision2DPy` as the working folder, and run
  * `a0_dtb.aa1_RT.a8_plot_nonparam_RT_recovery_MATLAB.py` to plot **supplements 1, 2, 4, and 5**
  * `a0_dtb.aa1_RT.a10_plot_model_comp_dtb_simple.py` to plot **supplement 3**
* To fit models anew, move parameter files in `Data_2D/Fit.D2.RT.Td2Tnd.Main/` elsewhere and run `main_fig2supp1_5.m`. (This may take days to weeks depending on your setup.)
* To fit with only a few iterations to see how the code runs, find `'MaxIter'` in `main_fig2supp1_5.m` and change it to a small number (e.g., 1).

(YK wrote this part of the code.)

## Figure 3
(AZ wrote this part of the code.)

## Figure 4
* In Python, set `Decision2DPy` as the working folder, and run `a0_dtb.aa2_VD.a2_dtb_2D_fit_VD.py` to load and use fitted parameters to reproduce the figure.
* To fit models anew, move parameter files in `Data_2D/Data_2D_Py/a0_dtb/VD/` elsewhere and run `a2_dtb_2D_fit_VD.py`. (This may take days to weeks depending on your setup.)
* To fit with only a few iterations to see how the code runs, find `'max_epoch'` in `a2_dtb_2D_fit_VD.py` and change it to a small number (e.g., 1).

(YK wrote this part of the code.)

## Figure 4 supplements 1 and 2
* Run `Decision2DPy/a0_dtb/aa2_VD/a2_dtb_2D_recover_VD.py` to load and use fitted parameters to reproduce the figure. Note that you need files in `Data_2D/Data_2D_Py/a0_dtb/VD/` to start this part of the analysis; otherwise the above fits will be run anew.
* To fit models anew, move parameter files in `Data_2D/Data_2D_Py/a0_dtb/VD_model_recovery/` elsewhere and run `a2_dtb_2D_recover_VD.py`. (This may take days to weeks depending on your setup.)
* To fit with only a few iterations to see how the code runs, find `'max_epoch'` in `a2_dtb_2D_fit_VD.py` and change it to a small number (e.g., 1). Note that this is done the same way as for Figure 4, since both analyses use the same code for fitting.

(YK wrote this part of the code.)

## Figure 5 and Figure 5 supplement 1
(AZ wrote this part of the code.)

## Figure 5 supplement 2
(AL worte this part of the code.)

## Figure 6
(AZ wrote this part of the code.)

## Figure 7 and Figure 7 supplement 1
(AL worte this part of the code.)
