#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from collections import OrderedDict as odict
from pprint import pprint

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import MultivariateNormal, Uniform, Normal

from a03_variable_dur import coef_by_dur_vs_odif as calc
from lib.pylabyk import np2
from lib.pylabyk import numpytorch as npt, yktorch as ykt
from lib.pylabyk.numpytorch import npy, npys
from lib.pylabyk import plt2, np2, localfile

from data_2d import consts, load_data


locfile = localfile.LocalFile(
    pth_root='../../Data_2D/Data_2D_Py/a01_RT/aa03_ch_ixn'
)


class Dtb1D(ykt.BoundedModule):
    def __init__(
            self,
            kappa0=10.,
            bound0=1.,
            diffusion=1.,
            y0=0.,
            ssq0=1e-6,
            dt=1/75,
            n_ev = 2 ** 7 + 1,
            max_ev = 3
    ):
        super().__init__()
        self.kappa = ykt.BoundedParameter(kappa0, 0.01, 50.)
        self.bias = ykt.BoundedParameter(0., -0.5, 0.5)
        self.bound = ykt.BoundedParameter(bound0, 0.1, 2.5)
        self.diffusion = ykt.BoundedParameter(diffusion, 0.99, 1.01)
        self.y0 = ykt.BoundedParameter(y0, -0.5, 0.5)
        self.ssq0 = ykt.BoundedParameter(ssq0, 1e-6, 1e-1)

        self.dt = dt
        assert n_ev % 2 == 1  # for padding in conv1d to work
        self.n_ev = n_ev
        self.max_ev = max_ev
        self.ev_bin = torch.linspace(-self.max_ev, self.max_ev, self.n_ev)
        self.dev = self.ev_bin[1] - self.ev_bin[0]

        self.max_ev_kernel = np.sqrt(diffusion * dt) * 3.5 + 0.5 * 50 * dt
        self.ev_bin_kernel = self.ev_bin[
            torch.abs(self.ev_bin) < self.max_ev_kernel
        ]

    def forward(self, ev):
        """

        @param ev: [condition, frame]
        @return: p_absorbed[condition, frame, ch]

        @type ev: torch.Tensor
        @rtype: torch.Tensor
        """
        nt = ev.shape[-1]
        n_cond = ev.shape[-2]
        batch_shape = ev.shape[:-2]

        pev = npt.sumto1(
            torch.exp(
                Normal(loc=self.y0.v,
                       scale=torch.sqrt(self.ssq0.v)).log_prob(
                    self.ev_bin
                )
            ).expand(
                batch_shape + torch.Size([n_cond] + [-1])
            ).unsqueeze(0), -1)

        ev = npt.p2st(ev)
        norm_kernel = Normal(loc=0.,
                             scale=torch.sqrt(self.diffusion.v * self.dt))
        ev_bin_kernel = self.ev_bin_kernel.expand(
            torch.Size([1] * (1 + len(batch_shape)) + [1, -1])
        )
        pad = ev_bin_kernel.shape[-1] // 2

        p_absorbed = torch.empty(
            torch.Size([nt])
            + batch_shape
            + torch.Size([n_cond, 2])
        )

        mask_abs = torch.stack([
            torch.clamp(
                (-self.bound.v - self.ev_bin) / self.dev,
                0., 1.
            ),  # mask_down
            torch.clamp(
                (self.ev_bin - self.bound.v) / self.dev,
                0., 1.
            )  # mask_up
        ], -1)  # [ch, ev]
        mask_in = (
            (1. - npt.p2st(mask_abs)[0])
            * (1. - npt.p2st(mask_abs)[1])
        )

        for t, ev1 in enumerate(ev):
            kernel = npt.sumto1(
                torch.exp(norm_kernel.log_prob(
                    (ev1[:, None, None] + self.bias.v)
                    * self.kappa.v * self.dt
                    + ev_bin_kernel
                )), -1)
            pev = F.conv1d(
                pev, kernel,
                groups=n_cond,
                padding=pad
            )
            a = torch.sum(
                pev.unsqueeze(-1) * mask_abs[None, None, :], -2
            ).squeeze(-3)  # [cond, ch]
            p_absorbed[t] = a
            pev = pev * mask_in[None, None, :]
            # print(p_absorbed[t].shape)
            # print('--')
        return npt.p2en(p_absorbed).transpose(-2, -1)  # [cond, fr, ch]


class Dtb2D(ykt.BoundedModule):
    pass


class Dtb2DSer(Dtb2D):
    pass


class Dtb2DPar(Dtb2D):
    pass


class Dtb2DInh(Dtb2D):
    pass


class Dtb2DTarg(Dtb2D):
    pass


if __name__ == '__main__':
    model = Dtb1D()
    ev = torch.arange(-5, 6)[:, None] + torch.zeros(1, 10)
    p_abs = model(ev)

    cost = torch.log(torch.sum(npt.p2st(p_abs)[0]))
    cost.backward()
    pprint({
        k: (v.v.data, v._param.grad) for k, v in model.named_modules() if
        k != ''
    })
    print(p_abs.shape)

    n_cond = ev.shape[0]
    colors = plt.get_cmap('cool', n_cond)

    nt = ev.shape[1]
    t = np.arange(nt) * model.dt

    n_row = 2
    for ch in range(2):
        for i, p_abs1 in enumerate(p_abs):
            plt.plot(
                t,
                npy(p_abs1[:, ch]) * np.sign(ch - 0.5),
                color=colors(i)
            )

    plt.show()