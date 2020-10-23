#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

#%%
from a0_dtb.aa1_RT import a9_plot_nonparam_RT_MATLAB as nonparam_rt
from a0_dtb.aa1_RT import a8_plot_nonparam_RT_recovery_MATLAB as \
    nonparam_rt_recover
from a0_dtb.aa2_VD import a2_dtb_2D_fit_VD as fit_VD
from a0_dtb.aa2_VD import a3_dtb_2D_recover_VD as recover_VD

nonparam_rt.main()
nonparam_rt_recover.main()
fit_VD.main()
recover_VD.main()