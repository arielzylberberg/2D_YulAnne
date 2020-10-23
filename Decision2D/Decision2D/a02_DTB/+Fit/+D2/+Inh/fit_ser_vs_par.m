%% Fit serial with constant bound and sigmaSq 1
W = Fit.D2.Inh.MainBatch;
W.batch_ixn_irr_ser('bound', 'Const', 'sigmaSq', 'Const');

%% Fit parallel with constant bound and sigmaSq 1
W = Fit.D2.Inh.MainBatch;
W.batch_ixn_irr_par('bound', 'Const', 'sigmaSq', 'Const');
