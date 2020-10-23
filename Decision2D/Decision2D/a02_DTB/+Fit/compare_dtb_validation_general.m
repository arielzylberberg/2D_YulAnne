function [ds_cost, file_ds] = compare_dtb_validation_general( ...
    S_comp, S_hold, S_const, varargin)
% Compare validation cost between S_comp within each S_hold, 
% with S_const constant.

S_opt = varargin2S(varargin, {
    'replace_file_field', struct
    });
S_opt.replace_file_field = varargin2S(S_opt.replace_file_field);

S_const = varargin2S(S_const, {
    'skip_existing_csv', true
    'force_to_use_easiest_only_for_comparison', []
    'parad', 'RT'
    });
assert(ischar(S_const.parad), 'parad must be char (not cell array)!');

% SS = varargin2S(varargin, {
%     'skip_existing_csv', true
%     'force_to_use_easiest_only_for_comparison', []
%     });

%% Load fit & compute validation cost
[Ss_hold, n_hold] = factorizeS(S_hold);
[Ss_comp, n_comp] = factorizeS(S_comp);

S_vary = varargin2S(S_hold, S_comp); % To name comparison results file
S_all = varargin2S(S_vary, S_const);
[Ss_all, n_all] = factorizeS(S_all);

%%
% switch S_hold.parad
%     case 'RT'
%         W0 = Fit.D2.Common.Main;
%         S_batch = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn(varargin{:});
%         
%     case 'sh'
%         W0 = Fit.D2.Common.Main;
%         S_batch = W0.get_S_batch_sh(varargin{:});
%         
%     otherwise
%         error('Unsupported parad:%s', S_hold.parad);
% end

%% (legacy)
% S0 = varargin2S(varargin, {
%     'parad', 'RT'
%     });
% switch S0.parad
%     case 'RT'
%         W0 = Fit.D2.Common.Main;
%         S_batch = W0.get_S_batch_fit_RT_Inh_BetaCdf_Const_Ixn(varargin{:});
%         
%     case 'sh'
%         W0 = Fit.D2.Common.Main;
%         S_batch = W0.get_S_batch_sh(varargin{:});
% end
% Ss = factorizeS(S_batch);
% 
% subjs = S_batch.subj(:)';
% n_subj = numel(subjs);
% 
% models = S_batch.model(:)';
% n_model = numel(models);

%% Construct a template object
S0 = Ss_all(1);
C = S2C(S0);
W0 = Fit.D2.Common.Main;
switch S0.parad
    case 'RT'
        W = W0.create_RT(C{:});
    case 'sh'
        W = W0.create_sh(C{:});
end

%% Name the result file
S2s = bml.str.Serializer;
S_name_ds = S2s.convert_to_S_file(...
    varargin2S(S_vary, S0), [
        W.file_fields
        {'tr_incl_prct', 'trp'}
        ]);
n_subj = length(S_name_ds.sbj);
% if n_subj > 3
S_name_ds.sbj = {S_name_ds.sbj{1}, sprintf('x%d', n_subj)};
if ~isempty(S_const.force_to_use_easiest_only_for_comparison)
    S_name_ds.ec = ...
        S_const.force_to_use_easiest_only_for_comparison;
end
% end
for f = fieldnames(S_opt.replace_file_field)'
    if isfield(S_name_ds, f{1})
        S_name_ds.(f{1}) = S_opt.replace_file_field.(f{1});
    end
end
file_ds = fullfile(W.get_dir, S2s.convert(S_name_ds));
% file_ds = S2s.convert(S2s.convert_to_S_file( ...
%     varargin2S(S_vary, W.S0_file), ...
%     W.file_fields));
fprintf('file_ds: %s\n', file_ds);

% C = varargin2C({
%     'subj', subjs{1}
%     'model', models{1}
%     }, Ss_all(1));
% switch S0.parad
%     case 'RT'
%         W = W0.create_RT(C{:});
%     case 'sh'
%         W = W0.create_sh(C{:});
% end
% file_ds = W.get_file({
%     'sbj', subjs
%     'mdl', models
%     });

%% Skip existing
if S_const.skip_existing_csv ...
        && exist([file_ds, '.mat'], 'file') ...
        && exist([file_ds, '.csv'], 'file')
    fprintf('Loading existing %s.mat\n', file_ds);
    L = load([file_ds, '.mat'], 'ds_cost');
    ds_cost = L.ds_cost;
    disp(ds_cost);
    return;
end

%% Name columns for each comparand
comps = cell(1, n_comp);
S2s = bml.str.Serializer;
for i_comp = 1:n_comp
    S_file = S2s.convert_to_S_file(Ss_comp(i_comp), W.file_fields);
    comps{i_comp} = bml.str.alphanumeric_name( ...
        S2s.convert(S_file));
end

disp('comps:');
disp(comps(:));

%% Load fits / find what is missing
cost = nan(n_hold, n_comp);
Ws = cell(n_hold, n_comp);
files = cell(n_hold, n_comp);
is_missing = false(n_hold, n_comp);
% to_break = false;

for i_comp = 1:n_comp
    for i_hold = 1:n_hold
        
%         S = Ss_all(1);
%         S.subj = subj;
%         S.model = model;

        S_comp1 = Ss_comp(i_comp);
        S_hold1 = Ss_hold(i_hold);
        S = varargin2S(varargin2S(S_comp1, S_hold1), S_const);
        C = S2C(S);
    
        try
            switch S0.parad
                case 'RT'
                    W = W0.create_RT(C{:});
                case 'sh'
                    W = W0.create_sh(C{:});
            end
        catch err
            warning('Error trying i_comp=%d, i_hold=%d:\n', i_comp, i_hold);
            disp(S);
            warning(err_msg(err));
            is_missing(i_hold, i_comp) = true;
            continue;
        end
        
        file = W.get_file;
        files{i_hold, i_comp} = file;
        
        if ~exist([file, '.mat'], 'file')
            warning('File missing: %s.mat\n', file);
            is_missing(i_hold, i_comp) = true;
%             to_break = true;
%             break;
%             keyboard;
        end        
    end
end

%% Fill in the cost
for i_comp = 1:n_comp
    for i_hold = 1:n_hold
        if ~is_missing(i_hold, i_comp)
            file = files{i_hold, i_comp};
            
            S_comp1 = Ss_comp(i_comp);
            S_hold1 = Ss_hold(i_hold);
            S = varargin2S(varargin2S(S_comp1, S_hold1), S_const);
            
            try
                L = load([file, '.mat']);
                fprintf('Loaded %s.mat\n', file);
                W1 = L.Fl.W;
                
                to_use_easiest_only0 = W1.to_use_easiest_only;
                if ~isempty(S.force_to_use_easiest_only_for_comparison)
                    W1.to_use_easiest_only = ...
                        S.force_to_use_easiest_only_for_comparison;
                else
                    W1.to_use_easiest_only = ...
                        W1.to_use_easiest_only_for_comparison;
                end
                cost1 = W1.get_cost;
                W1.to_use_easiest_only = to_use_easiest_only0;
                
                cost(i_hold, i_comp) = cost1;
%                 cost(i_subj, i_comp) = W1.get_cost_validation;
                Ws{i_hold, i_comp} = W1;
            catch err
                warning(err_msg(err));
                keyboard;
            end
        end
    end
end

%% Compare
fs_cost = fieldnames(Ss_hold);
n_col = numel(fs_cost);
n_row = n_hold;
C_cost = cell(n_row + 1, n_col);

for i_col = 1:n_col
    f1 = fs_cost{i_col};
    C_cost{1, i_col} = fs_cost{i_col};
    C_cost(2:end, i_col) = vVec({Ss_hold.(f1)});
end
ds_cost = cell2ds2(C_cost);

%% Human-readable part
[cost_best, i_best] = min(cost, [], 2);
ds_cost.cost_best = cost_best;
ds_cost.i_best = i_best;
ds_cost.best = vVec(comps(i_best));
dcost_best = bsxfun(@minus, cost, cost_best);

for i_comp = 1:n_comp
    comp = comps{i_comp};
    ds_cost.(['BFbest_' comp]) = exp(dcost_best(:, i_comp));
end

for i_comp = 1:n_comp
    comp = comps{i_comp};
    ds_cost.(['dbest_' comp]) = dcost_best(:, i_comp);
end

%% Data dump
for i_comp = 1:n_comp
    comp = comps{i_comp};
    ds_cost.(comp) = cost(:, i_comp);
    
%     if i_comp > 1
%         ds_cost.dcost(:, i_comp - 1) = ...
%             ds_cost.(comps{i_comp}) - ds_cost.(comps{1});
%     end
end

ds_cost.dcost_best = dcost_best;

%%
% ds_cost.dcost = ds_cost.(comps{2}) - ds_cost.(comps{1});
% ds_cost.BayesFactor = exp(ds_cost.dcost);
ds_cost.BayesFactorBest = exp(ds_cost.dcost_best);
ds_cost.files = files;

disp(ds_cost);
disp(ds_cost.best);
disp(ds_cost.dcost_best);

%%
export(ds_cost, 'File', [file_ds, '.csv'], 'Delimiter', ',');
save([file_ds, '.mat'], 'S_comp', 'S_hold', 'S_const', 'ds_cost');
fprintf('Saved to %s.csv and .mat\n', file_ds);
end