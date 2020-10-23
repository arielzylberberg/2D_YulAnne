#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

"""Fields:
         models: 2 list[str_]
          subjs: 3 list[str_]

                 [subj, model, frame, cond_M, cond_C, ch_M, ch_C]
          datas: (3, 1, 376, 9, 9, 2, 2) ndarray[float64]
          preds: (3, 2, 376, 9, 9, 2, 2) ndarray[float64]

                 [subj, model]
            ths: (3, 2) ndarray[odict]
       th_names: (3, 2, 45) ndarray[str]

                 # original file
          file0: (1,) ndarray[str_]

                 [frame, cond_M, cond_C, ch_M, ch_C]
  data_avg_subj: (376, 9, 9, 2, 2) ndarray[float64]
  pred_avg_subj: (376, 9, 9, 2, 2, 2) ndarray[float64]
"""

import numpy as np
from scipy.io import loadmat
from pprint import pprint
from importlib import reload
from collections import OrderedDict as odict

from lib.pylabyk import np2


file_matlab_fit = '../../Data_2D/Fit.CompareModels.main_plot_pooled_ser_vs_par.mat'
dat = loadmat(file_matlab_fit)
print('Loaded %s' % file_matlab_fit)

for k in ['data_avg_subj', 'pred_avg_subj']:
    dat[k] = dat[k].astype(np.float)

for k in ['models', 'subjs']:
    dat[k] = [v[0] for v in list(dat[k].flatten())]

for k in ['datas', 'preds']:
    v0 = dat[k]

    sh0 = v0.shape
    sh1 = v0[0, 0].shape
    # v = [v.astype(np.float) for v in v0.flatten()]
    v0 = np.stack([v.astype(np.float) for v in v0.flatten()]).reshape(
        sh0 + sh1
    )
    dat[k] = v0

sh0 = dat['th_values'].shape
dat['ths'] = np.array([
    odict([(k[0], v[0].astype(np.float))
           for (k, v) in zip(ks.flatten(), vs.flatten())])
    for ks, vs in zip(dat['th_names'].flatten(), dat['th_values'].flatten())
]).reshape(sh0)

for k in ['th_names', 'th_values']:
    v0 = dat[k]
    dat[k] = np.array([
        list([s[0] for s in v1.flatten()])
        for v1 in v0.flatten()
    ], dtype=np.object).reshape(v0.shape + v0[0, 0].shape[:1])
dat['th_values'] = dat['th_values'].astype(np.float)

np2.dict_shapes(dat)