#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from collections import OrderedDict as odict

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import MultivariateNormal, Uniform, Normal

from lib.pylabyk import np2
from lib.pylabyk import numpytorch as npt, yktorch as ykt
from lib.pylabyk.numpytorch import npy, npys
from lib.pylabyk import plt2, np2, localfile


if __name__ == '__main__':
    dt = 1/75
    n_cond = 6
    cond_shape = torch.Size([n_cond])

    n_ev = 2 ** 7 + 1
    ev_bin = torch.linspace(-1.2, 1.2, n_ev)
    dev = ev_bin[1] - ev_bin[0]

    pev = npt.sumto1(torch.exp(Normal(0., 1e-6).log_prob(ev_bin)).expand(
        cond_shape + torch.Size([-1])).unsqueeze(0), -1)

    ev_bin_kernel = ev_bin[
        torch.abs(ev_bin) <= np.sqrt(dt) * 3.5 + 0.5 * 50 * dt
    ].expand(
        cond_shape + torch.Size([1, -1]))

    kappa = torch.nn.Parameter(torch.tensor(50.))
    bound = torch.nn.Parameter(torch.tensor(0.3))
    bias = torch.nn.Parameter(torch.tensor(0.))
    drift = (torch.arange(n_cond) / 10. - bias) * dt * kappa
    ev_bin_kernel1 = ev_bin_kernel + drift.reshape([-1, 1, 1])

    kernel = npt.sumto1(
        torch.exp(Normal(0., np.sqrt(dt)).log_prob(
            ev_bin_kernel1
        )),
        -1
    )

    mask_up = torch.clamp((ev_bin - bound) / dev, 0., 1.)
    mask_down = torch.clamp((-bound - ev_bin) / dev, 0., 1.)
    mask_in = (1. - mask_up) * (1. - mask_down)

    nt = 10
    p_up = torch.empty(nt, n_cond)
    p_down = torch.empty(nt, n_cond)

    for t in range(nt):
        pev = F.conv1d(
            pev,  # [1, n_cond, n_ev]
            kernel,  # [n_cond, 1, n_ev_kernel]
            groups=n_cond,
            padding=ev_bin_kernel.shape[-1] // 2
        )
        p_up[t] = torch.sum(pev * mask_up[None, None, :], -1).squeeze(0)
        p_down[t] = torch.sum(pev * mask_down[None, None, :],
                              -1).squeeze(0)
        pev = pev * mask_in[None, None, :]

    # print(pev.shape)
    # print(kernel.shape)
    # print(pev.shape)

    cost = torch.log(torch.sum(p_up[0]))
    cost.backward()
    print(kappa.grad)
    print(bound.grad)
    print(bias.grad)

    colors = plt.get_cmap('cool', n_cond)

    n_row = 4
    plt.subplot(n_row, 1, 1)
    for i, (ev1, kernel1) in enumerate(zip(ev_bin_kernel, kernel)):
        plt.plot(*npys(ev1.T, kernel1.T), color=colors(i))
    plt.title('kernel')

    plt.subplot(n_row, 1, 2)
    # plt.plot(*npys(ev_bin, pev[0, 0, :]))
    for i, pev1 in enumerate(pev.squeeze(0)):
        plt.plot(*npys(ev_bin, pev1), color=colors(i))
    plt.title('P(ev)')

    plt.subplot(n_row, 1, 3)
    t_all = np.arange(nt) * dt
    for i, p_up1 in enumerate(p_up.T):
        plt.plot(*npys(t_all, p_up1), color=colors(i))
    plt.ylabel('P(t,up)')

    plt.subplot(n_row, 1, 4)
    t_all = np.arange(nt) * dt
    for i, p_down1 in enumerate(p_down.T):
        plt.plot(*npys(t_all, -p_down1), color=colors(i))
    plt.ylabel('P(t,down)')
    plt.xlabel('t (s)')

    plt.show()
