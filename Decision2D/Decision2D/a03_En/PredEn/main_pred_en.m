

% rename_data_BoundedEn; % Run once

%%
n_dim = 2;
subj = 2;
Fls = cell(1, n_dim);

%%
for dim_rel = 1:n_dim
    file = get_file_BoundedEn(subj, dim_rel);
    L = load(file);
    fprintf('Loaded %s\n', file);

    Fl = L.Fl;
    W = Fl.W;

    W.Data.en_fields{1} = 'mME';
    W.Dtb.Bound.init_params0;
    W.th = L.res.th;
    % For backward compatibility
    W.th.Dtb__Bound__b_logitmean = logit(L.res.th.Dtb__Bound__b_mean);
    W.pred;

    Fls{dim_rel} = Fl;
    
    figure(dim_rel);
    clf;
    W.plot_plotfuns;    
end


%%
PE = PredEn(Fls);

%% Then simulate 2D RevCor for serial, parallel, and buffered serial