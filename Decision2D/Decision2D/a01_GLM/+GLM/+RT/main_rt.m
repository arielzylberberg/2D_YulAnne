init_path;

%% Model comparison - under construction
% W0 = GLM.RT.Main;
% W0.batch({
%     });

%% Full model - RT
W0 = GLM.RT.MainFull;
W0.batch('parad', 'RT', 'subj', {'S1', 'S2', 'S3', 'FR'});

%% Full model - sh
W0 = GLM.RT.MainFull;
W0.batch('parad', 'sh', 'subj', {'S1', 'S2', 'S3'});