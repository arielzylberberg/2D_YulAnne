%%
init_path;

for subj = {'DX', 'MA', 'VL'} % , 'DX'}
    San = Fit.D2.Inh.SanityCheck('subj', subj{1});
    San.main;
end

%% Check InhSlice
% tbl = San.tbl_models;
% 
% get_W1 = @(s) tbl.W1{find(strcmp(tbl.model1, s), 1, 'first')};
% 
% Ser = get_W1('Ser');
% InhSlice = get_W1('InhSlice');
% InhSliceFree = get_W1('InhSliceFree');