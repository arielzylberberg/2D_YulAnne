clear;
init_path;

%% Common Setting
C0 = varargin2C({
    'subj', Data.Consts.subjs_RT(3)' % (:)'
    'parad', 'sh'
    'task', Data.Consts.tasks(1, 1) % (1,:)
    ... 'n_dim_task', 1
    ... 'dim_rel_W', dim_rel
    ...
    'fix_bias_st', false
    'fix_sigmaSq_st', false
    ...
    'drift_kind', 'Const' % {'Const', 'Exp'} % Try time-varying drift
    'bound_kind', 'BetaMeanAsymDec' % {'BetaMean', 'BetaMeanAsymDec', 'BMA2'}
    'kbratio_kind', 'n'
    'sigmaSq_kind', 'Const' % {'Const', 'LinearMinPreDrift'}
    'disper_kind', 'std'
    ...
    'skip_existing_mat', false
    'skip_existing_fig', false
    });

%% Run on cluster
W0 = Fit.D1.Bounded.Main;
W0.batch(C0{:});

%% Local testing
W = Fit.D1.Bounded.Main(C0{:});
W.main;

%% Compare models within each subject & task
[Ss_data, n_dataset] = factorizeS(varargin2S({
    'subj', Data.Consts.subjs_RT(:)'
    'parads', {'sh'}
    'task', Data.Consts.tasks(1,:)
    }));

W0 = Fit.D1.Bounded.Main;
for i_dataset = 1:n_dataset
    S_data1 = Ss_data(i_dataset);
    
    S1 = varargin2S(S_data1, C0);
    [Ss_model, n_model] = factorizeS(S1);
    
    for i_model = 1:n_model
        W = Fit.D1.Bounded.Main(C0{:});
        file = W.get_file;
        
    end
end