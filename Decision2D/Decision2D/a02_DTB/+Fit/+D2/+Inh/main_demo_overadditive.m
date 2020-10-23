%%
init_path;
W = Fit.D2.Inh.MainBatch;
W.pred;
W.plot_plotfuns

%%
W.th.Dtb__p_dim1_1st = 0.5;
W.th.Dtb__drift_fac_together_dim1_1 = 0.5;
W.th.Dtb__drift_fac_together_dim2_1 = 0.5;
W.th.Dtb__sigmaSq_fac_together_dim1_1 = 0.5;
W.th.Dtb__sigmaSq_fac_together_dim2_1 = 0.5;
W.th.Dtb__drift_fac_together_dim1_2 = 0.5;
W.th.Dtb__drift_fac_together_dim2_2 = 0.5;
W.th.Dtb__sigmaSq_fac_together_dim1_2 = 0.5;
W.th.Dtb__sigmaSq_fac_together_dim2_2 = 0.5;
W.pred;
W.plot_plotfuns;