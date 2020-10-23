function [ds_cost, file_ds] = compare_dtb_validation(varargin)
% Compare validation cost

SS = varargin2S(varargin, {
    'skip_existing_csv', true
    'force_to_use_easiest_only_for_comparison', []
    });

%% Load fit & compute validation cost
S0 = varargin2S(varargin, {
    'parad', 'RT'
    });
switch S0.parad
    case 'RT'
        W0 = Fit.D2.Common.Main;
        S_batch = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn(varargin{:});
        
    case 'sh'
        W0 = Fit.D2.Common.Main;
        S_batch = W0.get_S_batch_sh(varargin{:});
end
Ss = factorizeS(S_batch);

subjs = S_batch.subj(:)';
n_subj = numel(subjs);

models = S_batch.model(:)';
n_model = numel(models);

%% Skip existing
C = varargin2C({
    'subj', subjs{1}
    'model', models{1}
    }, Ss(1));
switch S0.parad
    case 'RT'
        W = W0.create_RT(C{:});
    case 'sh'
        W = W0.create_sh(C{:});
end
file_ds = W.get_file({
    'sbj', subjs
    'mdl', models
    });
if SS.skip_existing_csv ...
        && exist([file_ds, '.mat'], 'file') ...
        && exist([file_ds, '.csv'], 'file')
    fprintf('Loading existing %s.mat\n', file_ds);
    L = load([file_ds, '.mat'], 'ds_cost');
    ds_cost = L.ds_cost;
    disp(ds_cost);
    return;
end

%%
cost = nan(n_subj, n_model);
Ws = cell(n_subj, n_model);
files = cell(n_subj, n_model);
is_missing = false(n_subj, n_model);
% to_break = false;

for i_model = 1:n_model
    model = models{i_model};
    for i_subj = 1:n_subj
        subj = subjs{i_subj};
        
        S = Ss(1);
        S.subj = subj;
        S.model = model;
        C = S2C(S);
    
        switch S0.parad
            case 'RT'
                W = W0.create_RT(C{:});
            case 'sh'
                W = W0.create_sh(C{:});
        end
        file = W.get_file;
        files{i_subj, i_model} = file;
        
        if ~exist([file, '.mat'], 'file')
            warning('File missing: %s.mat\n', file);
            is_missing(i_subj, i_model) = true;
%             to_break = true;
%             break;
%             keyboard;
        end        
    end
end
for i_model = 1:n_model
    for i_subj = 1:n_subj
        if ~is_missing(i_subj, i_model)
            file = files{i_subj, i_model};
            try
                L = load([file, '.mat']);
                fprintf('Loaded %s.mat\n', file);
                W1 = L.Fl.W;
                
                to_use_easiest_only0 = W1.to_use_easiest_only;
                if ~isempty(SS.force_to_use_easiest_only_for_comparison)
                    W1.to_use_easiest_only = ...
                        SS.force_to_use_easiest_only_for_comparison;
                else
                    W1.to_use_easiest_only = ...
                        W1.to_use_easiest_only_for_comparison;
                end
                cost1 = W1.get_cost;
                W1.to_use_easiest_only = to_use_easiest_only0;
                
                cost(i_subj, i_model) = cost1;
%                 cost(i_subj, i_model) = W1.get_cost_validation;
                Ws{i_subj, i_model} = W1;
            catch err
                warning(err_msg(err));
                keyboard;
            end
        end
    end
end

%% Compare
ds_cost = cell2ds2([
    {'subj'}
    subjs(:)
    ]);
for i_model = 1:n_model
    model = models{i_model};
    ds_cost.(model) = cost(:, i_model);
    
    if i_model > 1
        ds_cost.dcost(:, i_model - 1) = ...
            ds_cost.(models{i_model}) - ds_cost.(models{1});
    end
end
% ds_cost.dcost = ds_cost.(models{2}) - ds_cost.(models{1});
ds_cost.BayesFactor = exp(ds_cost.dcost);
disp(ds_cost);

export(ds_cost, 'File', [file_ds, '.csv'], 'Delimiter', ',');
save([file_ds, '.mat'], 'S_batch', 'subjs', 'models', 'cost', 'ds_cost');
fprintf('Saved to %s.csv and .mat\n', file_ds);
end