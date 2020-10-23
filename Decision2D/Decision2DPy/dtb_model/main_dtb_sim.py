#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
from matplotlib import pyplot as plt
import scipy
from pprint import pprint

import torch

from lib.pylabyk import numpytorch as npt
from lib.pylabyk import decisionplot
from lib.pylabyk import plt2, np2
from lib.pylabyk.numpytorch import npy, npys

from dtb_model import dtb_sim
from data_2d import consts
from data_2d.consts import T_MAX

#%%
n_tr = 1000
dt = 1/25 # DEBUG
nt = int(T_MAX // dt)
t = np.arange(nt) * dt
n_dim = consts.N_DIM

model_kind = 'Ser'
if model_kind == 'Par':
    model = dtb_sim.Dtb2DParSim(dt=dt)
elif model_kind == 'Ser':
    model = dtb_sim.Dtb2DSerSim(dt=dt)
else:
    raise ValueError('Unsupported model_kind=%s' % model_kind)

# ev = (torch.tensor([0., 1.]) / 10.)[None, :, None] + torch.zeros(n_tr, 1, nt)
ev = npy(npt.normrnd(
    torch.tensor([0., 0.01]),
    np.sqrt(dt) / 10,
    (n_tr, nt)
).permute([0, 2, 1]))
t_kernel = np.arange(25) * dt
ev_kernel = scipy.stats.gamma.pdf(t_kernel, 1.99, scale=0.05)
# plt.plot(t_kernel, ev_kernel)
# plt.show()

for tr in range(n_tr):
    for dim in range(n_dim):
        ev[tr, dim, :] = np.convolve(ev[tr, dim, :], ev_kernel,
                                     mode='same')
# ev = scipy.signal.convolve(ev, ev_kernel)
# plt.plot(t, ev[0, 0, :])
# plt.show()

ev = torch.tensor(ev)
ch, rt, state = model.simulate(ev)
# log_p_ch_rt = dtb(ev, ch, rt)

# dim_on[trial, dim, time]
dim_on = state.dv_open

dplt = decisionplot.Decision(dt=dt)
n_col = 3
for dim in range(n_dim):
    plt2.subplotRC(n_dim, n_col, dim + 1, 1)
    dplt.hist_ch_rt(*npys(ch[:, dim], rt[:, dim]))

    plt2.subplotRC(n_dim, n_col, dim + 1, 2)
    dplt.ev_for_ch(*npys(ev[:,dim,:], ch[:, dim]),
                   summary_within_trial='None')
    plt.axhline(0, **consts.Style.axhlinestyle)

    plt2.subplotRC(n_dim, n_col, dim + 1, 3)
    plt.plot(t, dim_on[10:12, dim, :].T)

for col, title in enumerate([
    'P(RT, ch)', 'En for ch', 'Example DimOn'
]):
    plt2.subplotRC(n_dim, n_col, 1, col + 1)
    plt.title(title)

plt.show()
print('--')

#%%

pass