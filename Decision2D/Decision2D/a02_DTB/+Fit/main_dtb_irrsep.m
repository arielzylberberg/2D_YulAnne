% Fit 1D DTB separately on each irr condition.

%% ---- Common
clear;
init_path;
W0 = Fit.D2.IrrSep.Main;

%% ---- Settings
C_data = {
%     {
%     'n_dim_task', 2
%     'dim_rel_W', 1
%     }
    {
    'n_dim_task', 2
    'dim_rel_W', 2
    }
    };
C_group = {
%     {
%     'group_kind', 'dif_irr_incl'
%     'group_val', {{1:2, 4:5}}
%     'accu_irr_incl', 0:1
%     }
    {
    'group_kind', 'cond_irr_incl'
    'group_val', {{1:3, 4:6, 7:9}}
    'accu_irr_incl', 0:1
    }
    };
Cs = bml.args.factorize_merge_C({C_data, C_group});
C_model = {
        {
        'class_Ws', 'Fit.D1.Bounded.Main'
        'bound_kind', 'BetaMeanAsym'
        'sigmaSq_kind', 'Const' % 'LinearMinPreDrift'
        'fix_miss', false
        'fix_bias_st', false
        'fix_sigmaSq_st', false
        }
    };        
[Cs, n] = bml.args.factorize_merge_C( ...
    {Cs, C_model}, {
    % Custom options here
    'subj', Data.Consts.subjs_RT(3)
    });

%% ---- Run
for ii = 1:n
    C = Cs{ii};
    W0.batch(C{:});
end

%% ---- Imgather
init_path;
W0 = Fit.D2.IrrSep.Main;

%%
files = bml.str.clipboard2list;

%%
W0.batch_plot_files(files);

%%
% W0.batch_all_imgather_files;
for ii = 1:n
    Ss{ii} = varargin2S(Cs{ii});
end

%%
for ii = 1:n
    W0.batch_imgather_files(files, 'allof', Ss{ii});
end

%%
for ii = 1:n
    W0.batch_imgather_files_params(files, 'allof', Ss{ii});
end

