nam = {
    'DX', 'S1'
    'MA', 'S2'
    'VL', 'S3'
    };

%%
dir0 = cd;
cd('../Docs/Data_for_paper/En/DtbEn/Fit.D1.BoundedEn.Main');
for subj = 1:3
    bml.file.strrep_filename(nam{subj, 1}, nam{subj, 2});
end
cd(dir0);

%%
for subj = 1:3
    for dim_rel = 1:2
        file = get_file_BoundedEn(subj, dim_rel);
        L = load(file);
        fprintf('Loaded %s\n', file);
        
        pth = L.Fl.W.Data.path;
        pth = strrep(pth, nam{subj, 1}, nam{subj, 2});
        
        L.Fl.W.Data.subj = nam{subj, 2};
        L.Fl.W.Data.path = pth;

        L.W.Data.subj = nam{subj, 2};
        L.W.Data.path = pth;
        
        L.S0_file.subj = nam{subj, 2};
        
        disp(L.Fl.W.Data.path);
        disp(L.W.Data.path);
        
        save(file, '-struct', 'L');
        fprintf('Saved back to %s\n', file);
    end
end