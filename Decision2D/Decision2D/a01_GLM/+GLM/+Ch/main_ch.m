%% Model comparison
init_path;
W0 = GLM.Ch.Main;
W0.batch({
    'subj', {'S1'} % FR'}
    'tr_incl_prct', [0, 100] % [0, 25]
    });

%% Full model
init_path;
W0 = GLM.Ch.MainFull;
W0.batch({
    'subj', {'S1', 'S2', 'S3'} % {'FR'}
    'dim_rel_W', {1, 2}
    'n_dim_task', {2}
    'tr_incl_prct', [0, 100] % [0, 25] % [0, 50] 2 
    });

%%
L = load('../Data_2D/sTr/RT_S2.mat');
dat0 = L.dat;

%%
clf;
W = GLM.Ch.MainFull;
incl = dat0.task == 'A';
dat = dat0(incl, :);
W.plot_ch('dat', dat, 'dim_rel', 1);

%%
W0.batch_short;