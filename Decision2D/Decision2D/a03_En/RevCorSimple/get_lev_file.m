function [file, S0] = get_lev_file(spec, varargin)
S = varargin2S(varargin, {
    'pth', '../Data_2D/a03_En/RevCorSimple'
    });
n_subj = 3;
n_dim = 2;
dif_rels = {1, 2, 3};
dif_irrs = {1,2,3,4,5};
aligns = [-1, 1];
n_bin_to_pool = 12;

S0 = varargin2S(spec, {
    'subj', num2cell(1:n_subj)
    'n_dim_task', num2cell(1:n_dim)
    'dim_rel', num2cell(1:n_dim)
    'dif_rel', dif_rels
    'dif_irr', dif_irrs
    'align', num2cell(aligns)
    'n_bin_to_pool', n_bin_to_pool
    });
nam = bml.str.Serializer.convert(S0);
file = fullfile(S.pth, nam);
end