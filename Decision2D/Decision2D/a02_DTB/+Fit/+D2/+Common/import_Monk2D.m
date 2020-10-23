function Dat = import_Monk2D(ds, varargin)
S = varargin2S(varargin, {
    'subj', Data.Consts.subjs_RT{1}
    'parad', 'VD'
    'task', 'A'
    });

fs = {
    'cond_M', 'condM'
    'ch_M',   'subjM'
    'cond_C', 'condC'
    'ch_C',   'subjC'
    'ans_M',  'corrM'
    'ans_C',  'corrC'
    };
nf = size(fs, 1);

for ii = 1:nf
    ix = strcmp(ds.Properties.VarNames, fs{ii, 1});
    
    assert(nnz(ix) == 1, 'Column %s is absent!\n', fs{ii, 1});
    ds.Properties.VarNames{ix} = fs{ii, 2};
end
n_tr = size(ds, 1);
ds.task = repmat('A', [n_tr, 1]);
ds.RT = ds.RT(:,1);
ds.subjM = ds.subjM + 1;
ds.subjC = ds.subjC + 1;

C = S2C(S);
Dat = Fit.D2.Common.DataChRtPdf(C{:});
Dat.set_ds0(ds);
% Dat.set_ds(ds);
Dat.filt_ds;
end