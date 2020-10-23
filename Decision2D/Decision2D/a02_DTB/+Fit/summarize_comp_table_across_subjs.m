function ds_subj = summarize_comp_table_across_subjs(files)
%%
% # comparisons
n_comp = numel(files);
file1 = files{1};

L = load(file1);
subjs = unique(L.subjs);
cost1 = L.cost;

n_subj = size(cost1, 1);
n_model = size(cost1, 2);

models = L.models(1:n_model);

fval_model = nan(n_comp, n_model, n_subj);
fval_sum = nan(n_comp, n_model);
model_best = cell(n_comp, 1);

for i_comp = 1:n_comp
    file1 = files{i_comp};
    
    %%
    L = load(file1);
    fprintf('Loaded %s\n', file1);
    
    %%
    % fval_model(comparison, model, subj)
    % <- cost(subj, model)
    fval_model(i_comp, :, :) = ...
        permute(L.cost, [3, 2, 1]);
end

%%
% summarize across comparisons
fval_sum = permute(nanmean(fval_model, 1), [3, 2, 1]); 

% min across models for each subject
[fval_best, ix_model_best] = min(fval_sum, [], 2); 

dfval_sum = bsxfun(@minus, fval_sum, fval_best);
model_best = models(ix_model_best);

disp(dfval_sum);
disp(model_best);

%% 
% ds_subj: summarize per subject
ds_subj = dataset;

end