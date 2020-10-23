%%
files = csprintf('Inh_par%02d', [1, 3, 5, 8, 10]);
files = [
    files(:); {
    'InhSigma_drift010_5.000000e-01'
    'InhSigma_drift050'
    'InhSigma_drift010'
    }];

%%
n_files = numel(files);
for ii = 1:n_files
    file = files{ii};
    
    load([file '.mat'], 'res');
    fval(ii) = res.fval;
    
    fprintf('%s: %10.1f\n', file, fval(ii));
end
