%% Batch
init_path;
W0 = EnIxn.GLM.MainLags;
C = varargin2C({
    'subj', {'DX', 'MA', 'VL'}
    ... 'dif_incl', 1:2
    'dim_rel_W', 1
    'dif_rel_incl', {1} % {1,2}
    'dif_irr_incl', {1} % {1,2}
    't0_rel', {'st', 'en'} % {'st', 'en'}
    't0_opp', true
    'smooth_ms', 100
    'lev_kind', 'bsame'
    'n_shuf', 400
    'skip_existing_mat', true
    });
W0.batch(C{:});    

%% Test
init_path;
W = EnIxn.GLM.MainLags;
C = varargin2C({
    'subj', 'MA'
    'dif_rel_incl', 1 % 1:2, ...
    'dif_irr_incl', 1 % 1:2, ...
    't0_kind', 'st'
    'smooth_ms' 100
    'n_shuf', 400
    'lev_kind', 'bsame'
    });
W.init(C{:});
%%
disp(W.load_if_existing);

%%
W.imagesc;

%%
W.main;

% W.calculate;
% clf;
% W.imagesc;


