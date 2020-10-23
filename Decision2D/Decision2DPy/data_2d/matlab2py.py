#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
from scipy.io import loadmat
import h5py
import os
import pprint
import pandas as pd

#%%
from lib.pylabyk import zipPickle as zpkl
from lib.pylabyk.matlab2py import unpackarray, structlist2df
from lib.pylabyk.np2 import dict_shapes

#%%
def asc2str(v):
    return np.squeeze(np.array(v, dtype=np.uint8)).tostring().decode('ascii')

#%%
def load_eye_all():
    #%% Load data
    print(os.getcwd())

    pth = '../Data_2D/sTr/'
    nam = 'sTr_eye_all_subj_parad_py'
    file_full = pth + nam + '.mat'
    # fit0 = loadmat(file_full, struct_as_record=False)
    f = h5py.File(file_full, 'r')
    print('Loaded ' + file_full)

    #%%
    d = {k:np.squeeze(np.array(f[k])) for k in f.keys() if k[0] != '#'}

    #%%
    pprint.pprint({k:d[k] for k in d.keys()})

    #%%
    for k in d.keys():
        if d[k].dtype == np.dtype('O'):
            print('Converting ' + k)
            d[k] = [np.squeeze(np.array(f[v])) for v in d[k]] # keep as lists
    print('Conversion done!')

    #%%
    return d

    #%% Saving to zpkl takes a very long time
    # pkl_file = pth + nam + '.zpkl'
    # zpkl.save(d, pkl_file)
    # print('Saved to ' + pkl_file)
    #
    # #%% Test
    # dat = zpkl.load(pkl_file)
    # dict_shapes(dat)

#%%

