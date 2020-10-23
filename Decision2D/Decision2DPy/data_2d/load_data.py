#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
from scipy.io import loadmat

from lib.pylabyk import np2
from lib.pylabyk.np2 import filt_dict

from data_2d import consts

#%%
def ____LOAD____():
    pass

scale_en = [1/600, 1/4] # to make them similar to the coherence

file_matlab_combined = '../../data/orig/data_RT_VD.mat'
# file_matlab_combined = '../../Data_2D/sTr/combined_2D_RT_sh_VD_unibimanual' \
#                        '.mat'
dat = loadmat(file_matlab_combined)
print('Loaded %s' % file_matlab_combined)
for key in ['parads', 'subjs']:
    d = dat[key]
    d2 = [s[0] for s in d.flatten()]
    # for ii in range(d.shape[0]):
    #     d1 = d[ii][0]
    #     if len(d1) == 0:
    #         d1 = ''
    #     else:
    #         d1 = d1[0]
    #     d2.append(d1)
    dat[key] = d2

for key in dat.keys():
    if key.startswith('id_'):
        dat[key] -= 1  # MATLAB starts with 1; Python starts with 0

for key, val in dat.items():
    if (isinstance(val, np.ndarray)
            and val.ndim == 2
            and val.shape[1] == 1):
        dat[key] = val.flatten()

# # REMOVE: en unused
# dat['en'] = np.transpose(dat['en'], [0, 2, 1])
# dat['en'] *= np2.vec_on(np.array(scale_en), 1, 3)

dat['dim_rel'] = dat['dim_rel'].astype(np.bool)
dat['to_excl'] = dat['to_excl'].astype(np.bool)

# REMOVE: S1 filtered already
# # Exclude S1's last two blocks of A, where they guessed on motion
# dat1 = np2.filt_dict(dat, (
#     dat['id_subj'] == dat['subjs'].index('S1')
# ) & (
#     dat['task'] == 'A'
# ) & (
#     dat['id_parad'] == dat['parads'].index('RT')
# ))
# runs_to_exclude = np.unique(dat1['i_all_Run'])[-2:]
#
# dat1_high_motion_excl = np2.filt_dict(
#     dat1, (dat1['cond'][:, 0] == 0.512)
#           & np.isin(dat1['i_all_Run'], runs_to_exclude)
# )
#
# dat1_high_motion_incl = np2.filt_dict(
#     dat1, (dat1['cond'][:, 0] == 0.512)
#           & ~np.isin(dat1['i_all_Run'], runs_to_exclude)
# )

# # Plot showing why last two runs of S1 are an exception:
# from matplotlib import pyplot as plt
# for dim in [0, 1]:
#     accu = []
#     se_accu = []
#     i_runs = np.unique(dat1['i_all_Run'])
#     for i_run in i_runs:
#         dat2 = np2.filt_dict(dat1, (
#             (np.abs(dat1['cond'][:, dim]) > 0)
#             & (dat1['i_all_Run'] == i_run)
#         ))
#         accu1 = (np.sign(dat2['cond'][:, dim]) ==
#                  np.sign(dat2['ch'][:, dim] - 1.5))
#         accu.append(np.mean(accu1))
#         se_accu.append(np2.sem(accu1))
#     plt.errorbar(i_runs, accu, yerr=se_accu)
# plt.show()

# REMOVE: already filtered
# # --- Remove last two runs of S1, during which the motion accuracy was at chance
# dat1 = np2.filt_dict(dat, ~np.array((
#     dat['id_subj'] == dat['subjs'].index('S1')
# ) & np.isin(dat['i_all_Run'], runs_to_exclude) & (
#     dat['id_parad'] == dat['parads'].index('RT')
# )
# ))
# dat = dat1

# # REMOVE: color converted already
# #  --- Change color coherence values for VD and manual
# # color_coh = 'logit'  # '2p_blue-1' | 'logit'
# color_coh = '2p_blue-1'
# for parad in ['VD', 'RT']:
# # for parad in ['VD', 'RT', 'sh']:  # REMOVE: sh unused
#     for subj in consts.SUBJS[parad]:
#         incl = (
#             (dat['id_parad'] == dat['parads'].index(parad))
#             & (dat['id_subj'] == dat['subjs'].index(subj))
#         )
#         dim_color = consts.DIM_NAMES_SHORT.index('C')
#         coh_color = dat['cond'][incl, dim_color]
#
#         # # REMOVE: color coh already unified
#         # if color_coh == '2p_blue-1':
#         #     if parad == 'VD':
#         #         coh_color = coh_color * 2
#         #     elif parad in ['RT', 'sh']:
#         #         coh_color = 2 * np2.logistic(coh_color) - 1.
#         #
#         # elif color_coh == 'logit':
#         #     sign_coh = np.sign(coh_color)
#         #     _, ix_abs_coh = np.unique(np.abs(coh_color), return_inverse=True)
#         #     for i_abs_coh in range(np.amax(ix_abs_coh) + 1):
#         #         incl1 = ix_abs_coh == i_abs_coh
#         #         coh_color[incl1] = sign_coh[incl1] * consts.COHS_COLOR[parad][subj][
#         #             i_abs_coh]
#         # else:
#         #     raise ValueError()
#
#         dat['cond'][incl, dim_color] = coh_color

# NOTE: unibimanual is left unchanged at this point

def load_data_combined():
    return dat


def load_data_parad(parad='VD'):
    raise NotImplementedError()

#%%
def ___MOMENTARY_EVIDENCE____():
    pass

file_impulse = '../Data_2D/sTr/impulse_MotionEnergy.csv'
def load_impulse():
    impulse = np.loadtxt(file_impulse)
    return impulse