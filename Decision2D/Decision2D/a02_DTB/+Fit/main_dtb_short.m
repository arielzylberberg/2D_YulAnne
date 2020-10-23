% main_dtb_short
% Estimate length of the buffer

clear;
init_path;
global to_use_parallel


%% Inh - Settings - Common
% global to_use_parallel
init_path;

to_use_easiest_only = {1};

C0 = varargin2C({
    'model', {'InhSlice', 'InhPar'} % {'InhSliceInv'} % {'InhSlice', 'InhPar', 'InhSliceFree', 'Ser', 'Par'} % {'InhSer', 'InhPar', 'InhSlice', 'InhSliceFree'}
    'parad', 'sh'
    'drift_kind', 'IrrSep' % Exp' % Exp' % Exp' % {'IrrSep', 'IrrSepExp'}
    'bound_kind', 'BetaMeanAsymDec' % 'BetaMean' % AsymDec' % Collapses to 0 % Decreasing
    'sigmaSq_kind', 'LinearMinPreDrift' % 'Const' % 'Const' % 'LinearMinPreDrift'; %'Const'|'LinearMinPreDrift'
    'miss_kind', '' % 'Avg'|''
    'tnd_kind', 'invgauss'
    'fix_irr_ixn', false % true % 
    'fix_sigmaSq_st', false
    'fix_drift_t_st', true
    'fix_bias_st', false
    ...
    'to_excl_outlier_runs', false % true
    'skip_existing_mat', false %
    'skip_existing_fig', false
    'UseParallel', 'always' % 'never' % 
    ... 'MaxIter', 0
    'to_use_easiest_only', to_use_easiest_only
    'to_use_easiest_only_for_fit', to_use_easiest_only
    'to_use_easiest_only_for_comparison', ...
        cellfun(@(v) -v, to_use_easiest_only, 'UniformOutput', false)
    ... Short paradigm specific
    'p_dim1_1st', 1
    'fix_p_dim1_1st', true % false
    'buffer_dur_sec', 0.12 - 4/75
%     'b_mean0', 0.3 % 
    });

%% Run on cluster
to_use_parallel = true; %#ok<NASGU>
C = varargin2C({
    'subj', Data.Consts.subjs_RT(1) % [3,2,1])'
    'UseParallel', 'always' % 'always'|'never'
    'MaxIter', 1e4
    }, C0);
W0 = Fit.D2.Common.Main;
W0.batch_fit_sh(C{:});
% W0.batch_plot_sh(C{:});
% W0.batch_fit_RT_Inh_BetaCdf_Const_Ixn(C{:});

%% Compare cost
difs = to_use_easiest_only;
n_dif = numel(difs);
ds_costs = cell(1, n_dif);
file_dss = cell(1, n_dif);
for i_dif = 1:n_dif
    dif = difs{i_dif};
    C = varargin2C({
        'subj', Data.Consts.subjs_RT(:)'
        ...
        'to_use_easiest_only_for_fit', dif
        'to_use_easiest_only_for_comparison', -dif
        }, C0);
    
    [ds_costs{i_dif}, file_dss{i_dif}] = ...
        Fit.compare_dtb_validation(C{:});
end

%% Local testing
to_use_parallel = false;
C = varargin2C({
    'subj', Data.Consts.subjs_RT{3}
    'UseParallel', 'never' % 'always'|'never'
    'MaxIter', 0
    }, C0);
W0 = Fit.D2.Common.Main;

S = W0.get_S_batch_sh(C{:});
% S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn(C{:});
Ss = factorizeS(S);

W = W0.create_sh(Ss(1));
    
%%
disp(W.get_cost);
% W.plot_plotfuns;
W.get_Fl;
clf;
W.Fl.runPlotFcns;

%%
W.main;


%% Inh - Settings - Common
% global to_use_parallel
init_path;

to_use_easiest_only = {1:2}; % {0, 1:2, 1:3}; % , [1,2], [1,3]}; % 0;

C = varargin2C({
    'model', {'InhPar', 'InhSlice', 'InhSliceFree', 'Ser', 'Par'} % {'InhSer', 'InhPar', 'InhSlice', 'InhSliceFree'}
    'drift_kind', 'IrrSepExp' % Exp' % {'IrrSep', 'IrrSepExp'}
    'bound_kind', 'BetaMean' % AsymDec' % Collapses to 0 % 'BetaMeanAsymDec' % Decreasing
    'miss_kind', '' % 'Avg'|''
    'tnd_kind', 'invgauss'
    'fix_irr_ixn', false % true % 
    'fix_sigmaSq_st', false
    ...
    'to_excl_outlier_runs', true
    'skip_existing_mat', false %
    'skip_existing_fig', false
    'UseParallel', 'always' % 'never' % 
    ... 'MaxIter', 0
    'to_use_easiest_only', to_use_easiest_only
    'to_use_easiest_only_for_fit', to_use_easiest_only
    'to_use_easiest_only_for_comparison', ...
        cellfun(@(v) -v, to_use_easiest_only, 'UniformOutput', false)
    ... Short paradigm specific
    'p_dim1_1st', 1
    'fix_p_dim1_1st', true % false
    'buffer_dur_sec', 0.12 - 4/75
    });

%% Run on cluster
to_use_parallel = true; %#ok<NASGU>
C = varargin2C({
    'subj', Data.Consts.subjs_RT(3)'
    'UseParallel', 'always' % 'always'|'never'
    'MaxIter', 1e4
    }, C);
W0 = Fit.D2.Common.Main;
W0.batch_fit_sh(C{:});
% W0.batch_plot_sh(C{:});
% W0.batch_fit_RT_Inh_BetaCdf_Const_Ixn(C{:});

%% Local testing
to_use_parallel = false;
C = varargin2C({
    'subj', Data.Consts.subjs_RT{1}
    'UseParallel', 'never' % 'always'|'never'
    'MaxIter', 0
    }, C);
W0 = Fit.D2.Common.Main;

S = W0.get_S_batch_sh(C{:});
% S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn(C{:});
Ss = factorizeS(S);

W = W0.create_sh(Ss(1));
    
%%
disp(W.get_cost);
% W.plot_plotfuns;
W.get_Fl;
clf;
W.Fl.runPlotFcns;

%%
W.main;


%%
%% Ser & Par - Settings - Common
for bat = {
%         1, 'IrrSepExp'
%         [1,3], 'IrrSepExp'
%         [1,4], 'IrrSepExp'
%         [1,5], 'IrrSepExp'
        1:2, 'IrrSepExp'
%         [1,3], 'IrrSep'
%         [1,4], 'IrrSep'
%         [1,5], 'IrrSep'
        }'
    [dif, drift_kind] = deal(bat{:});

    
    C = varargin2C({
        'model', {'InhPar'} % , 'Par'} % , 'InhSlice', 'InhSliceFree'} % {'Par', 'Ser'} % {'InhSlice'} % {'InhSer'} % , 'InhSer', 'InhPar'} % 'InhDrift', 'InhFree'}, ...
        'drift_kind', drift_kind % 'IrrSepExp'
        'bound_kind', 'BetaMeanAsymDec' % Collapses to 0 % 'BetaMeanAsymDec' % Decreasing
        'miss_kind', '' % 'Avg'|''
        'tnd_kind', 'invgauss'
        'fix_irr_ixn', true
        'fix_sigmaSq_st', false
        'to_excl_outlier_runs', true
        'skip_existing_mat', false % true
        'skip_existing_fig', false % true
        'UseParallel', 'always' % 'never' % 
        ... 'MaxIter', 0
        'to_use_easiest_only', dif
        'to_use_easiest_only_for_fit', dif
        'to_use_easiest_only_for_comparison', -dif;
        ... Short paradigm specific
        'p_dim1_1st', 1
        'fix_p_dim1_1st', true % false
        'buffer_dur_sec', 0.12 - 4/75
        });

    %% Run on cluster
    to_use_parallel = true; %#ok<NASGU>
    C = varargin2C({
        'subj', Data.Consts.subjs_RT(:)'
        'UseParallel', 'always' % 'always'|'never'
        'MaxIter', 1e4
        }, C);
    W0 = Fit.D2.Common.Main;
    W0.batch_fit_sh(C{:});
    % W0.batch_fit_RT_Inh_BetaCdf_Const_Ixn(C{:});
end

%% Local testing
to_use_parallel = false;
C = varargin2C({
    'subj', Data.Consts.subjs_RT{2}
    'UseParallel', 'never' % 'always'|'never'
    'MaxIter', 0
    }, C);
W0 = Fit.D2.Common.Main;

S = W0.get_S_batch_sh(C{:});
% S = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn(C{:});
Ss = factorizeS(S);

W = W0.create_sh(Ss(2));
    
%%
disp(W.S0_file);
disp(W.get_file);
disp(W.get_cost);
% W.plot_plotfuns;
W.get_Fl;
clf;
W.Fl.runPlotFcns;

%%
W.main;

%% Param table
ds = res2table(L.res, ...
    'names', {
        'Dtb__', ''
        'log10', ''
        'logit', ''
        'log', ''
        '__', '_'
        '__', '_'
        }, ...
    'transform', {
        });
disp(ds);

%% == Compare cost
%% Common settings
init_path;
C0 = varargin2C({
    'subj', Data.Consts.subjs_RT
    'parad', 'sh'
    ... 'model', {'Ser', 'InhSliceFree'} % , 'InhSliceFree'} % {'Ser', 'Par'} % {'Ser', 'Par'} % {'Ser', 'Par'} % , 'InhSer', 'InhPar'} % {'Ser'}, ... % , 'Par', 'InhFree'}, ...
    ... 'drift_kind', 'IrrSepExp' % {'IrrSep', 'IrrSepExp'}
    'bound_kind', 'BetaMeanAsymDec' % Collapses to 0 % 'BetaMeanAsymDec' % Decreasing
    'miss_kind', '' % 'Avg'|''
    'tnd_kind', 'invgauss'
    'fix_irr_ixn', false
    'fix_sigmaSq_st', false
    ... Short paradigm specific
    'p_dim1_1st', 1
    'fix_p_dim1_1st', true % false
    'buffer_dur_sec', 0.12 - 4/75    
    ...
    'to_excl_outlier_runs', true
    'skip_existing_csv', true %
    ... 'skip_existing_fig', false
    ... 'UseParallel', 'always' % 'never' % 
    ... 'MaxIter', 0
    ... 'to_use_easiest_only_for_fit', dif
    ... 'to_use_easiest_only_for_comparison', -dif
    });

%% == Ser vs Par 
difs = {0}; % {-4, -5}; % {-1, -2, -3}; % [1,2], [1,3], [1,4], [1,5]}; % {1, 2, 3, [1,2], [1,3]};
n_dif = numel(difs);
ds_costs = cell(1, n_dif);
file_dss = cell(1, n_dif);
for i_dif = 1:n_dif
    dif = difs{i_dif};
    C = varargin2C({
        'model', {'Ser', 'Par'} % , 'InhSliceFree'} % {'Ser', 'Par'} % {'Ser', 'Par'} % {'Ser', 'Par'} % , 'InhSer', 'InhPar'} % {'Ser'}, ... % , 'Par', 'InhFree'}, ...
        'drift_kind', 'IrrSep' % {'IrrSep', 'IrrSepExp'}
        ...
        'to_use_easiest_only_for_fit', dif
        'to_use_easiest_only_for_comparison', -dif
        }, C0);
    
    [ds_costs{i_dif}, file_dss{i_dif}] = ...
        Fit.compare_dtb_validation(C{:});
end

%% == Ser vs Par vs InhSlice vs InhSliceFree 
difs = {1:2}; % {-1, -2, -3, -4, -5}; % {0}; % [1,2], [1,3], [1,4], [1,5]}; % {1, 2, 3, [1,2], [1,3]};
n_dif = numel(difs);
ds_costs_inh = cell(1, n_dif);
file_dss_inh = cell(1, n_dif);
for i_dif = 1:n_dif
    dif = difs{i_dif};
    C = varargin2C({
        'model', {'Ser', 'Par', 'InhSlice', 'InhSliceFree'} % , 'InhSliceFree'} % {'Ser', 'Par'} % {'Ser', 'Par'} % {'Ser', 'Par'} % , 'InhSer', 'InhPar'} % {'Ser'}, ... % , 'Par', 'InhFree'}, ...
        'drift_kind', 'IrrSepExp' % {'IrrSep', 'IrrSepExp'}
        ...
        'to_use_easiest_only_for_fit', dif
        'to_use_easiest_only_for_comparison', -dif
        }, C0);
    
    [ds_costs_inh{i_dif}, file_dss_inh{i_dif}] = ...
        Fit.compare_dtb_validation(C{:});
end

%% Summarize across comparisons
%% sh including dif1
files = {
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par,InhSlice,InhSliceFree}+dft=SE+bnd=AD+ssq=C+tnd=i+ntnd=4+msf=0+ef=[1,5]+ec=[-1,-5]+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par,InhSlice,InhSliceFree}+dft=SE+bnd=AD+ssq=C+tnd=i+ntnd=4+msf=0+ef=[1,4]+ec=[-1,-4]+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par,InhSlice,InhSliceFree}+dft=SE+bnd=AD+ssq=C+tnd=i+ntnd=4+msf=0+ef=[1,3]+ec=[-1,-3]+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par,InhSlice,InhSliceFree}+dft=SE+bnd=AD+ssq=C+tnd=i+ntnd=4+msf=0+ef=[1,2]+ec=[-1,-2]+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    };

%% RT, 5x cross-validation
files = {
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=RT+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par,InhSlice,InhSliceFree}+dft=S+bnd=A+ssq=C+tnd=i+ntnd=4+msf=0+ef=-1+ec=1+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=RT+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par,InhSlice,InhSliceFree}+dft=S+bnd=A+ssq=C+tnd=i+ntnd=4+msf=0+ef=-2+ec=2+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=RT+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par,InhSlice,InhSliceFree}+dft=S+bnd=A+ssq=C+tnd=i+ntnd=4+msf=0+ef=-3+ec=3+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=RT+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par,InhSlice,InhSliceFree}+dft=S+bnd=A+ssq=C+tnd=i+ntnd=4+msf=0+ef=-4+ec=4+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=RT+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par,InhSlice,InhSliceFree}+dft=S+bnd=A+ssq=C+tnd=i+ntnd=4+msf=0+ef=-5+ec=5+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    };

%% sh, 5x cross-validation
files = {
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par,InhSlice,InhSliceFree}+dft=SE+bnd=AD+ssq=C+tnd=i+ntnd=4+msf=0+ef=-1+ec=1+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par,InhSlice,InhSliceFree}+dft=SE+bnd=AD+ssq=C+tnd=i+ntnd=4+msf=0+ef=-2+ec=2+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par,InhSlice,InhSliceFree}+dft=SE+bnd=AD+ssq=C+tnd=i+ntnd=4+msf=0+ef=-3+ec=3+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par,InhSlice,InhSliceFree}+dft=SE+bnd=AD+ssq=C+tnd=i+ntnd=4+msf=0+ef=-4+ec=4+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par,InhSlice,InhSliceFree}+dft=SE+bnd=AD+ssq=C+tnd=i+ntnd=4+msf=0+ef=-5+ec=5+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    };

%% sh, Ser vs Par, 5x cross-validation
files = {
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par}+dft=S+bnd=AD+ssq=C+tnd=i+ntnd=4+msf=0+ef=-1+ec=1+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par}+dft=S+bnd=AD+ssq=C+tnd=i+ntnd=4+msf=0+ef=-2+ec=2+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par}+dft=S+bnd=AD+ssq=C+tnd=i+ntnd=4+msf=0+ef=-3+ec=3+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par}+dft=S+bnd=AD+ssq=C+tnd=i+ntnd=4+msf=0+ef=-4+ec=4+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    '../Data_2D/Fit.D2.IrrIxn.Main/sbj={DX,MA,VL}+prd=sh+tsk=A+dtk=2+dmr=1+trm=201+eor=1+mdl={Ser,Par}+dft=S+bnd=AD+ssq=C+tnd=i+ntnd=4+msf=0+ef=-5+ec=5+lf=0+td=Ser+fsqs=0+fbst=1.mat'
    };

%%
ds_subj = Fit.summarize_comp_table_across_subjs(files);
disp(ds_subj);