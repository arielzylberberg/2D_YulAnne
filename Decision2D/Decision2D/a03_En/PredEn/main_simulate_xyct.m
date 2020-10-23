% main_simulate_xyct

%%
SE = SimulateXyct;
L_RDK = SE.load_RDK;

%%
parad = 'RT';
for subj = 1:3
    file_in = fullfile('../Data_2D/sTr', ...
        sprintf('%s_S%d.mat', parad, subj));
    L1 = load(file_in);
    fprintf('Loaded %s\n', file_in);

    sTr = L1.dat;
    n_tr = size(sTr, 1);

    %%
    xycts = cell(n_tr, 1);
    success_sim_en = false(n_tr, 1);
    t_st = tic;
    fprintf('Simulating %d trials started at %s\n', ...
        n_tr, datestr(now, 'yyyymmddTHHMMSS'));
    parfor tr = 1:n_tr
        success1 = false;
        try
            SE = SimulateEn;
            SE.init(L_RDK);
            S1 = dataset2struct(sTr(tr, :));

            xyct = SE.simulate_xyct(S1.condM, invLogit(S1.condC), S1.seedM, S1.seedC);
            max_dif = SE.compare_xyct(S1.xyct, xyct);

            if any(max_dif > 1e-6)
                fprintf('Error in trial %d: max_dif: ', tr);
                disp(max_dif);
                success1 = false;
            else
                success1 = true;
            end
        catch err
            fprintf('Error in trial %d:\n', tr);
            warning(err_msg(err));
        end

        xycts{tr} = xyct;   
        success_sim_en(tr) = success1;

        if mod(tr, 10) == 0
            fprintf('.');
        end
        if mod(tr, 100) == 0
            fprintf('%d\n', tr);
        end
    end
    fprintf('%d done in %1.0f sec!\n', n_tr, toc(t_st));

    %%
    file_out = fullfile('../Data_2D/sTr', ...
        sprintf('%s_S%d_dot5s.mat', parad, subj));
    L2 = L1;
    L2.dat.xyct = xycts;
    L2.dat.success_sim_en = success_sim_en;
    save(file_out, '-struct', 'L2');
    
    info = dir(file_out);
    fprintf('Saved to %s (%1.0f MB)\n', file_out, info.bytes / 1e6);
end