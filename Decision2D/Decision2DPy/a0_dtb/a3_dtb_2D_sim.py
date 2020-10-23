#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
import time
import numpy_groupies as npg
from collections import OrderedDict as odict
from typing import Union, Iterable, Sequence, Callable, Tuple, Any

import torch
from torch.nn import functional as F
from torch import nn

from lib.pylabyk import numpytorch as npt, yktorch as ykt
from lib.pylabyk.numpytorch import npy
from lib.pylabyk import argsutil, plt2, np2

npt.device0 = torch.device('cpu')
ykt.default_device = torch.device('cpu')

from a0_dtb import a1_dtb_1D_sim as sim1d
from a0_dtb.a1_dtb_1D_sim import TimedModule
from a0_dtb.a1_dtb_1D_sim \
    import simulate_p_cond__rt_ch, rt_sec2fr
from data_2d.consts import get_kw_plot
from data_2d import consts

# run "tensorboard --logdir=runs" in the terminal to see log


def ____Utils____():
    pass


def get_demo_ev(n_cond=9, nt=consts.NT) -> torch.Tensor:
    levels = 0.5 * np.exp(-np.log(2)
                          * np.arange(n_cond // 2))
    levels = set(-levels).union(set(levels))
    if n_cond % 2 == 1:
        levels = levels.union({0.})
    levels = torch.tensor(np.sort(list(levels), 0))

    # levelss[cond12, dim]
    levelss = torch.stack([
        v.flatten() for v in torch.meshgrid([levels, levels])
    ], -1)

    # ev[cond12, fr, dim]
    ev = levelss[:, None, :].expand(-1, nt, -1)
    return ev


def unique_conds(cond_by_dim):
    """
    :param cond_by_dim: [tr, dim]
    :return: ev_conds[cond, dim], dcond[tr]
    """
    ev_conds, dcond = np.unique(cond_by_dim, return_inverse=True, axis=0)
    return ev_conds, dcond


class Data2D:
    def __init__(
            self,
            nt0=consts.NT,
            dt0=consts.DT,
            subsample_factor=1,
            **kwargs
    ):
        """

        :param nt0:
        :param dt0:
        :param subsample_factor:
        :param kwargs:
        """
        self.nt0 = nt0
        self.dt0 = dt0
        self.subsample_factor = subsample_factor
        self.n_ch = consts.N_CH_FLAT

        # Placeholders
        self.ev_cond_dim = np.empty([0, 0], dtype=np.float)  # type: np.ndarray
        self.ch_tr_dim = np.empty([0, 0], dtype=np.int)  # type: np.ndarray
        self.dcond_tr = np.empty([0], dtype=np.int)  # type: np.ndarray

    @property
    def nt(self):
        return int(self.nt0 // self.subsample_factor)

    @property
    def dt(self):
        return self.dt0 * self.subsample_factor

    @property
    def n_dcond0(self):
        return self.ev_cond_dim.shape[0]

    @property
    def n_tr0(self):
        return self.ch_tr_dim.shape[0]

    def get_incl(self, mode, fold_valid, mode_train, n_fold_valid,
                 i_fold_test=0, n_fold_test=1,
                 to_debug=False):

        in_tr = np.ones(self.n_tr0, dtype=np.bool)
        in_dcond = np.ones(self.n_dcond0, dtype=np.bool)
        in_tr_train_valid = np.ones(self.n_tr0, dtype=np.bool)

        if mode == 'all':
            pass

        elif mode in ['train', 'valid', 'train_valid', 'test']:
            # --- Choose training + validation set using mode_train
            def is_in_train_valid(mode_train):
                if mode_train == 'all':
                    if n_fold_test > 1:
                        in_tr_train_valid = (np.floor(
                            (np.arange(self.n_tr0) / self.n_tr0) * n_fold_test
                        ) != i_fold_test)
                    else:
                        in_tr_train_valid = np.ones(self.n_tr0, dtype=np.bool)

                    in_dcond_train_valid = np.ones(
                        len(self.ev_cond_dim), dtype=np.bool)

                elif mode_train == 'easiest':
                    if n_fold_test > 1:
                        raise ValueError()

                    ev_cond_dim = self.ev_cond_dim
                    dcond_tr = self.dcond_tr

                    easiest_cond_in_dim = np.abs(ev_cond_dim).max(0)
                    in_dcond_train_valid = np.any(
                        np.abs(ev_cond_dim) == easiest_cond_in_dim[None, :],
                        axis=1)

                    ix_dcond_train_valid = np.nonzero(in_dcond_train_valid)[0]
                    in_tr_train_valid = (
                            dcond_tr[:, None]
                            == ix_dcond_train_valid[None, :]
                    ).any(1)  # noqa

                    if to_debug:  # CHECKED mode_train == 'easiest'
                        print("==== Checking mode_train == 'easiest'")
                        print('ev_cond included:')
                        dcond_tr_incl0 = np.unique(dcond_tr[in_tr_train_valid])
                        print(ev_cond_dim[in_dcond_train_valid, :])
                        print(ev_cond_dim[dcond_tr_incl0, :])
                        does_ev_cond_agree = np.all(
                            ev_cond_dim[in_dcond_train_valid, :]
                            == ev_cond_dim[dcond_tr_incl0, :]
                        )
                        print('Does ev_cond agree with expected: %d'
                              % does_ev_cond_agree)
                        assert does_ev_cond_agree

                        print('easiest_in_dim:')
                        print(easiest_cond_in_dim)

                        n_cond_incl0 = np.sum(in_dcond_train_valid)
                        print('len(dcond_incl0): %d' % n_cond_incl0)
                        n_cond_incl1 = (
                                len(np.unique(ev_cond_dim[:, 0])) * 2
                                + len(np.unique(ev_cond_dim[:, 1])) * 2 - 4
                        )
                        print('# dcond along the boundary of the matrix: %d'
                              % n_cond_incl1)
                        does_n_cond_incl_agree = (n_cond_incl0 == n_cond_incl1)
                        print('Does len(dcond_incl0) agree with expected: %d'
                              % does_n_cond_incl_agree)
                        assert does_n_cond_incl_agree

                        print('====')
                else:
                    raise ValueError()
                return in_tr_train_valid, in_dcond_train_valid

            in_tr_train_valid, in_dcond_train_valid = is_in_train_valid(
                mode_train)

            # -- Choose test set by flipping training + validation set
            if mode == 'test':
                if mode_train == 'all':
                    if n_fold_test > 1:
                        in_tr = ~in_tr_train_valid
                    else:
                        in_tr = in_tr_train_valid
                    in_dcond = in_dcond_train_valid
                else:
                    in_tr = ~in_tr_train_valid
                    in_dcond = ~in_dcond_train_valid
            elif mode == 'train_valid' or n_fold_valid == 1:
                in_tr = in_tr_train_valid
                in_dcond = in_dcond_train_valid
            else:  # mode in ['train', 'valid']
                in_dcond = in_dcond_train_valid

                assert n_fold_valid > 1

                # get in_tr_valid
                n_tr = in_tr_train_valid.sum()
                ix_tr = np.zeros(self.n_tr0,
                                 dtype=np.long) + np.nan
                ix_tr[in_tr_train_valid] = np.arange(n_tr)
                in_tr_valid = np.floor(
                    ix_tr / n_tr * n_fold_valid
                ) == fold_valid

                if mode == 'valid':
                    in_tr = in_tr_valid
                else:  # mode == 'train':
                    in_tr = ~in_tr_valid & in_tr_train_valid

        else:
            raise ValueError()

        if to_debug:  # CHECKED n_fold_valid
            if mode != 'all':
                print('==== Checking n_fold_valid')
                n_tr_incl = in_tr.sum()
                n_tr_incl0 = in_tr_train_valid.sum()  # noqa
                n_tr = len(in_tr_train_valid)
                print(
                    '#tr_total: %d\n'
                    '#in_tr_train_valid: %d (%1.1f%% of all)\n'
                    '#in_tr: %d (%1.1f%% of tr_incl0)\n'
                    'fold_valid: %d/%d, mode: %s, mode_train: %s\n'
                    '===='
                    % (
                        n_tr,
                        n_tr_incl0, n_tr_incl0 / n_tr * 100,
                        n_tr_incl, n_tr_incl / n_tr_incl0 * 100,
                        fold_valid, n_fold_valid, mode, mode_train
                    )
                )
            plt.subplot(1, 2, 1)
            if mode != 'all':
                plt.plot(in_tr_train_valid, 'yo')
            plt.plot(in_tr, 'k.')
            plt.xlabel('trial')
            plt.title('Included in %s fold %d/%d'
                      % (mode, fold_valid, n_fold_valid))

            plt.subplot(1, 2, 2)
            plt.plot(*self.ev_cond_dim.T, 'yo')
            plt.plot(*self.ev_cond_dim[in_dcond].T, 'k.')
            plt.xlabel(consts.DIM_NAMES_LONG[0])
            plt.ylabel(consts.DIM_NAMES_LONG[1])
            plt.title('Included in %s among %s' % (mode, mode_train))
            plt.show()
            print('====')

        return in_dcond, in_tr, in_tr_train_valid


class Data2DRT(Data2D):
    def __init__(
            self,
            ev_tr_dim: np.ndarray,
            ch_tr_dim: np.ndarray,
            rt_tr: np.ndarray,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.ev_tr_dim = ev_tr_dim
        self.ch_tr_dim = ch_tr_dim
        self.rt_tr = rt_tr
        self.n_ch = consts.N_CH_FLAT

        self.ev_cond_dim = np.empty(())
        self.dcond_tr = np.empty(())
        self.update_data()
        # # self.n_cond_rt_ch, self.ev_cond_fr_dim_meanvar,
        # self.ev_cond_dim, self.dcond_tr = self.dat2p_dat(
        #     ch_tr_dim, rt_tr, ev_tr_dim)[2:4]

    def simulate_data(self, pPred_cond_rt_ch: torch.Tensor, seed=0,
                      rt_only=False):
        torch.random.manual_seed(seed)

        if rt_only:
            dcond_tr = self.dcond_tr
            chSim_tr_dim = self.ch_tr_dim
            chSimFlat_tr = consts.ch_by_dim2ch_flat(chSim_tr_dim)
            pPred_tr_rt = pPred_cond_rt_ch[dcond_tr, :, chSimFlat_tr]
            rtSim_tr = npy(npt.categrnd(
                probs=npt.sumto1(pPred_tr_rt, -1)
            ) * self.dt)
        else:
            dcond_tr = self.dcond_tr
            pPred_tr_rt_ch = pPred_cond_rt_ch[dcond_tr, :, :]
            n_tr, nt, n_ch = pPred_tr_rt_ch.shape
            chSim_tr_rt_ch = npy(npt.categrnd(
                probs=pPred_tr_rt_ch.reshape([n_tr, -1])))
            rtSim_tr = npy((chSim_tr_rt_ch // n_ch) * self.dt)
            chSim_tr = npy(chSim_tr_rt_ch % n_ch)

            chs = np.array(consts.CHS)
            chSim_tr_dim = np.stack([
                chs[dim][chSim_tr]
                for dim in range(consts.N_DIM)
            ], -1)

        self.update_data(ch_tr_dim=chSim_tr_dim, rt_tr=rtSim_tr)

    def update_data(
            self,
            ch_tr_dim: np.ndarray = None,
            rt_tr: np.ndarray = None,
            ev_tr_dim: np.ndarray = None
    ):
        if ch_tr_dim is None:
            ch_tr_dim = self.ch_tr_dim
        else:
            self.ch_tr_dim = ch_tr_dim

        if rt_tr is None:
            rt_tr = self.rt_tr
        else:
            self.rt_tr = rt_tr

        if ev_tr_dim is None:
            ev_tr_dim = self.ev_tr_dim
        else:
            self.ev_tr_dim = ev_tr_dim

        self.ev_cond_dim, self.dcond_tr = self.dat2p_dat(
            npy(ch_tr_dim), npy(rt_tr), ev_tr_dim)[2:4]

    def get_data_by_cond(
            self, mode='all', i_fold_valid=0, epoch=0,
            mode_train='all', n_fold_valid=1,
            i_fold_test=0, n_fold_test=1,
            upsample_ev=1,
            to_debug=False
    ) -> (torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray):
        """

        :param mode: 'all'|'train'|'valid'|'train_valid'|'test'
        :param i_fold_valid:
        :param epoch:
        :param mode_train: 'all'|'easiest'
        :param n_fold_valid:
        :param upsample_ev: 1 to disable upsampling; ~5 looks nice
        :param to_debug:
        :return: ev_cond_fr_dim_meanvar, n_cond_rt_ch, in_tr, in_dcond, \
               ev_cond_dim
        """
        in_dcond, in_tr, in_tr_train_valid = self.get_incl(
            mode, i_fold_valid, mode_train, n_fold_valid,
            i_fold_test=i_fold_test, n_fold_test=n_fold_test,
            to_debug=to_debug
        )

        n_cond_rt_ch, ev_cond_fr_dim_meanvar, ev_cond_dim, dcond_tr = \
            self.dat2p_dat(
                self.ch_tr_dim[in_tr],
                self.rt_tr[in_tr],
                self.ev_tr_dim[in_tr],
            )

        ev_cond_dim = self.ev_cond_dim[in_dcond]
        return ev_cond_fr_dim_meanvar, n_cond_rt_ch, in_tr, in_dcond, \
            ev_cond_dim

    def dat2p_dat(
        self,
        ch_tr_dim: np.ndarray,
        rt_sec: np.ndarray,
        ev_tr_dim: np.ndarray,
    ) -> (torch.Tensor, torch.Tensor, np.ndarray, np.ndarray):
        """
        :param ch_tr_dim: [tr, dim]
        :param rt_sec: [tr]
        :param ev_tr_dim: [tr, dim]
        :return: n_cond_rt_ch[cond, rt, ch],
        ev_cond_fr_dim_meanvar[dcond, fr, dim, (mean, var)],
        ev_cond_dim[dcond, dim], dcond_tr[tr]
        """
        nt0 = self.nt0
        dt0 = self.dt0
        n_ch_flat = self.n_ch
        subsample_factor = self.subsample_factor

        nt = int(nt0 // subsample_factor)
        dt = dt0 * subsample_factor

        drt = rt_sec2fr(rt_sec=rt_sec, dt=dt, nt=nt)
        ch_flat = consts.ch_by_dim2ch_flat(ch_tr_dim)

        ev_cond_dim, dcond_tr = unique_conds(ev_tr_dim)
        n_cond_flat = len(ev_cond_dim)
        ev_cond_fr_dim = torch.tensor(ev_cond_dim)[:, None, :].expand(
            [-1, nt, -1])

        ev_cond_fr_dim_meanvar = torch.stack([
            ev_cond_fr_dim, torch.zeros_like(ev_cond_fr_dim)
        ], -1)

        n_cond_rt_ch = torch.tensor(npg.aggregate(
            np.stack([dcond_tr, drt, ch_flat.astype(np.long)]),
            1., 'sum', [n_cond_flat, nt, n_ch_flat]
        ))
        return n_cond_rt_ch, ev_cond_fr_dim_meanvar, ev_cond_dim, dcond_tr


def upsample_ev(ev_cond_fr_dim_meanvar: torch.Tensor,
                dim_rel: int,
                steps=51  # CHECKING: revert to 51
                ) -> torch.Tensor:

    ev0 = ev_cond_fr_dim_meanvar

    ev_dim_cond = ev0[:, 0, :, 0].T
    evs_dim_cond = [v.unique() for v in ev_dim_cond]

    dim_irr = consts.get_odim(dim_rel)
    ev_rel = torch.linspace(
        evs_dim_cond[dim_rel].min(), evs_dim_cond[dim_rel].max(), steps=steps
    )
    ev_irr = evs_dim_cond[dim_irr]
    ev_rel, ev_irr = torch.meshgrid([ev_rel, ev_irr])
    ev = torch.stack([v.flatten() for v in [ev_rel, ev_irr]], -1)
    if dim_rel == 1:
        ev = ev.flip(-1)

    ev = ev[:, None, :].expand([-1, ev0.shape[1], -1])
    ev = torch.stack([ev, torch.zeros_like(ev)], -1)
    return ev


def ____Model_Classes____():
    pass


class Dtb2DRT(TimedModule):
    kind = 'None'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, ev: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Note that cond is ravelled across both dims, as is necessary to
        model, e.g., the other-dim-dependent bias.

        :param ev: [condition, frame, dim]
        :return: p_absorbed[condition, frame, ch]
        """
        raise NotImplementedError()

    def expand_ev(self, ev: torch.Tensor) -> torch.Tensor:
        """
        :param ev: [cond, fr, dim, (mean, var)] or [cond, dim]
        :return: ev[cond, fr, dim, (mean, var)]
        """
        if ev.ndim == 4:
            return ev
        elif ev.ndim == 2:
            ev_cond_dim_meanvar = torch.stack([
                ev, torch.zeros_like(ev)
            ], -1)
            ev_cond_fr_dim_meanvar = ev_cond_dim_meanvar.unsqueeze(1).expand(
                [-1, self.nt, -1, -1]
            )
            return ev_cond_fr_dim_meanvar
        else:
            raise ValueError("ev dimensions must be [cond, fr, dim, (mean, "
                             "var)] or [cond, dim]")


class Dtb2DRTDimensionWise(Dtb2DRT):
    kind = 'dimwise'

    def __init__(
            self,
            dtb1ds: Union[sim1d.Dtb1D, Sequence[sim1d.Dtb1D]] = sim1d.Dtb1D,
            to_allow_irr_ixn=False,
            **kwargs
    ):
        """
        :param dtb1ds:
        :param kwargs:
        """
        super().__init__(**kwargs)

        if type(dtb1ds) is type:
            kw = argsutil.kwdef(kwargs, {
                'timer': self.timer
            })
            dtb1ds = [
                dtb1ds(**kw)
                for dim in range(consts.N_DIM)
            ]
        self.dtb1ds = nn.ModuleList(dtb1ds)  # type: nn.ModuleList[sim1d.Dtb1D]

        self.to_allow_irr_ixn = to_allow_irr_ixn
        if to_allow_irr_ixn:
            self.kappa_rel_odim = ykt.BoundedParameter(
                [0., 0.], -0.1, 0.1
            )
            self.kappa_rel_abs_odim = ykt.BoundedParameter(
                [0., 0.], -0.1, 0.1
            )
        else:
            self.kappa_rel_odim = ykt.BoundedParameter(
                [0., 0.], 0., 0.
            )
            self.kappa_rel_abs_odim = ykt.BoundedParameter(
                [0., 0.], 0., 0.
            )

            # self.kappa_odim = ykt.BoundedParameter(
            #     [0., 0.], -0.1, 0.1
            # )
            # self.kappa_abs_odim = ykt.BoundedParameter(
            #     [0., 0.], -0.1, 0.1
            # )

    def ev_bins(self):
        return [dtb.ev_bin for dtb in self.dtb1ds]

    # def forward(self, ev: torch.Tensor, **kwargs
    #             ) -> Tuple[torch.Tensor, torch.Tensor]:
    #     """
    #     :param ev: [cond, fr, dim, (mean, var)]
    #     :return: p_cond__ch_td[cond, ch_flat, td],
    #     unabs_dim_td_cond_ev[dim, td, cond, ev]
    #     where ch_dim = self.chs[dim, ch_flat]
    #     """
    #     ev = self.expand_ev(ev)
    #
    #     # p_dim_cond__ch_td[dim, cond, ch, td] = P(ch_dim, td_dim | cond)
    #     p_dim_cond__ch_td = torch.stack([
    #         dtb.forawrd(ev1).permute([0, 2, 1])
    #         for ev1, dtb in zip(npt.p2st(ev), self.dtb1ds)
    #     ])
    #     p_dim_cond__ch_td_par = torch.stack([
    #         F.interpolate(p_td0,  # p_td0[cond, ch, fr]
    #                       p_td0.shape, 1./speed,
    #                       mode='linear')
    #         for speed, p_td0 in zip(self.speed_par, p_dim_cond__ch_td)
    #     ])
    #
    #     # return p_cond_td_ch

    def get_out_dtb1ds(self, ev: torch.Tensor, return_unabs: bool):
        """

        :param ev: [cond, fr, dim, (mean, var)]
        :param return_unabs:
        :return: p_dim_cond_td_ch, unabs_dim_td_cond_ev
        """

        ev1 = ev.clone()
        if self.to_allow_irr_ixn:
            for dim in range(consts.N_DIM):
                odim = consts.get_odim(dim)
                kappa = self.dtb1ds[dim].kappa[:]
                # ko = self.kappa_odim[odim] / kappa
                # kao = self.kappa_abs_odim[odim] / kappa

                okappa = self.dtb1ds[odim].kappa[:]
                ko = self.kappa_rel_odim[odim] / kappa * okappa
                kao = self.kappa_rel_abs_odim[odim] / kappa * okappa

                # ko = self.kappa_odim[odim]
                # kao = self.kappa_abs_odim[odim]

                ev1[:, :, dim, 0] = (
                    ev[:, :, dim, 0]
                    + ev[:, :, odim, 0] * ko
                    + ev[:, :, odim, 0].abs() * kao
                )
                # ev[:, :, dim, 1] = (
                #     ev0[:, :, dim, 1]
                #     + (ev0[:, :, odim, 1] * ko) ** 2
                #     + ((
                #         ev0[:, :, odim, 1]
                #         # + ev0[:, :, odim, 0] ** 2
                #         # - ev0[:, :, odim, 0].abs() ** 2  # see NOTE
                #     ) * kao) ** 2
                # )
                # NOTE: in fact we need to subtract mean(abs(ev)) instead of
                #  abs(mean(ev)), so this is
                #  incorrect when the evidence ever changes sign within the
                #  subsampled time step. The best way around this is to give
                #  subsample_factor directly to the Dtb module, and cache the
                #  subsampled evidence to avoid additional time cost.
                #  See: https://stats.stackexchange.com/a/89909/121823

        outs = [
            dtb(ev11, return_unabs=return_unabs)
            for ev11, dtb in zip(ev1.permute([2, 0, 1, 3]), self.dtb1ds)
        ]
        if return_unabs:
            p_dim_cond_td_ch = torch.stack([v[0] for v in outs])
            unabs_dim_td_cond_ev = torch.stack([v[1] for v in outs])
        else:
            # p_dim_cond__ch_td[dim, cond, ch, td] = P(ch_dim, td_dim | cond)
            p_dim_cond_td_ch = torch.stack(outs)
            unabs_dim_td_cond_ev = None
        return p_dim_cond_td_ch, unabs_dim_td_cond_ev


class Dtb2DRTSpeed(Dtb2DRTDimensionWise):
    kind = 'speed'

    def __init__(
            self,
            speed_par0=(0.5, 0.5),
            speed_par_lb=(0.01, 0.01),
            speed_par_ub=(1., 1.),
            freeze_speed=(False, False),
            **kwargs
    ):
        """
        Ser: Just freeze one of the dim's speed_par at 1, the other at 0.
        Par: Just freeze both dims' speed_par at 1.
        :param speed_par0:
        :param speed_par_lb:
        :param speed_par_ub:
        :param freeze_speed:
        :param kwargs:
        """
        super().__init__(**kwargs)

        self.speed_par = nn.ModuleList([
            ykt.BoundedParameter(th0, lb, ub)
            for dim, (th0, lb, ub)
            in enumerate(zip(speed_par0, speed_par_lb, speed_par_ub))
        ])
        for dim, frz in enumerate(freeze_speed):
            if frz:
                npt.freeze(self.speed_par[dim])


class Dtb2DRTSerial(Dtb2DRTDimensionWise):
    """
    """
    kind = 'serial'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, ev: torch.Tensor, return_unabs=False, **kwargs
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ev: [cond, fr, dim, (mean, var)]
        :return:
        """
        ev = self.expand_ev(ev)
        p_dim_cond_td_ch, unabs_dim_td_cond_ev = self.get_out_dtb1ds(
            ev, return_unabs)
        p_cond_td_ch = self.get_p_cond_td_ch(p_dim_cond_td_ch)
        return p_cond_td_ch, unabs_dim_td_cond_ev

    @staticmethod
    def get_p_cond_td_ch(p_dim_cond_td_ch: torch.Tensor) -> torch.Tensor:
        p_dim_cond_ch_td = p_dim_cond_td_ch.permute([0, 1, 3, 2])
        nt = p_dim_cond_ch_td.shape[-1]
        n_cond = p_dim_cond_ch_td.shape[1]
        p_cond_ch_td = torch.empty(n_cond, consts.N_CH_FLAT, nt)
        for ch_flat, ch_dims in enumerate(consts.CHS_TENSOR.T):
            p_cond_ch_td[:, ch_flat, :] = npt.conv_t(
                p_dim_cond_ch_td[0, :, ch_dims[0], :].unsqueeze(0),
                p_dim_cond_ch_td[1, :, ch_dims[1], :].unsqueeze(1),
                groups=n_cond
            )
        # # CHECKED
        # plt.imshow(npy(p_cond__ch_td[:, 3, :].sum([1])).reshape([9, 9]))
        # plt.colorbar()
        # plt.show()
        p_cond_td_ch = p_cond_ch_td.permute([0, 2, 1])
        return p_cond_td_ch


class Dtb2DRTParallel(Dtb2DRTDimensionWise):
    """
    """
    kind = 'parallel'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, ev: torch.Tensor, return_unabs=False, **kwargs
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param ev: [cond, fr, dim, (mean, var)] or [cond, dim]
        :return: p_cond_td_ch[cond, td, ch],
        unabs_dim_td_cond_ev[dim, td, cond, ev]
        """
        ev = self.expand_ev(ev)
        p_dim_cond_td_ch, unabs_dim_td_cond_ev = self.get_out_dtb1ds(
            ev, return_unabs)
        p_cond_td_ch = self.get_p_cond_td_ch(p_dim_cond_td_ch)
        return p_cond_td_ch, unabs_dim_td_cond_ev

    @staticmethod
    def get_p_cond_td_ch(p_dim_cond_td_ch: torch.Tensor) -> torch.Tensor:
        p_dim_cond_ch_td = p_dim_cond_td_ch.permute([0, 1, 3, 2])
        nt = p_dim_cond_ch_td.shape[-1]
        n_cond = p_dim_cond_ch_td.shape[1]
        p_cond_ch_td = torch.empty(n_cond, consts.N_CH_FLAT, nt)
        for ch_flat, ch_dims in enumerate(consts.CHS_TENSOR.T):
            out = npt.max_distrib(
                p_dim_cond_ch_td[
                    [0, 1], :, ch_dims[[0, 1]], :
                ].transpose(1, 2)
            )[0]

            p_cond_ch_td[:, ch_flat, :] = out.transpose(0, 1)

            # # CHECKED
            # cond12 = 2 * 9 + 4
            # ps = []
            # for dim1, color in zip([0, 1], ['r', 'b']):
            #     p1 = p_dim_cond__ch_td[dim1, cond12, ch_dims[dim1], :]
            #     ps.append(p1)
            #     plt.plot(npy(npt.sumto1(p1)), color)
            # p2 = p_cond__ch_td[cond12, ch_flat, :]
            # plt.plot(npy(npt.sumto1(p2)), 'k--')
            # plt.show()
            #
            # print(
            #     (ps[0].sum(-1) * ps[1].sum(-1) - p2.sum(-1)).abs().max()
            # )
            #
        # # CHECKED
        # plt.imshow(npy(p_cond__ch_td[:, 3, :].sum([1])).reshape([9, 9]))
        # plt.colorbar()
        # plt.show()
        p_cond_td_ch = p_cond_ch_td.permute([0, 2, 1])
        return p_cond_td_ch


class Dtb2DRTTargetwise(Dtb2DRT):
    pass


class FitRT2D(TimedModule):
    """
    Put DTB, Tnd, and Lase together to fit the data.
    """
    def __init__(
            self,
            dtb2d: Dtb2DRT = Dtb2DRTSerial,
            tnds: Union[sim1d.Tnd, Sequence[sim1d.Tnd]] = sim1d.TndLogNorm,
            lapse: sim1d.LapseUniform = sim1d.LapseUniform,
            to_normalize_within_cond=None,
            **kwargs
    ):
        super().__init__(**kwargs)

        if 'sumto1_wi_cond' in kwargs:
            assert to_normalize_within_cond is None, \
                'give either sumto1_wi_cond or to_normalize_within_cond, ' \
                'but not both!'
            to_normalize_within_cond = kwargs['sumto1_wi_cond']
        elif to_normalize_within_cond is None:
            to_normalize_within_cond = True

        self.chs = torch.tensor(consts.CHS)  # [dim, ch_flat] = ch_dim

        if type(dtb2d) is type:
            dtb2d = dtb2d(**{**kwargs, 'timer': self.timer})
        self.dtb = dtb2d

        n_tnd = consts.N_CH_FLAT
        if type(tnds) is type:
            tnds = [
                tnds(**{**kwargs, 'timer': self.timer})
                for i in range(n_tnd)
            ]
        else:
            assert len(tnds) == n_tnd
        self.tnds = nn.ModuleList(tnds) # the order is the same as chs

        if type(lapse) is type:
            lapse = lapse(**{**kwargs, 'timer': self.timer})
        self.lapse = lapse

        self.to_normalize_within_cond = to_normalize_within_cond

    def forward(self, ev: torch.Tensor):
        """

        :param ev: [condition, frame, dim, (mean, var)]
        :return: p_cond__rt_ch[condition, frame, ch_flat]
        """
        nt = ev.shape[1]
        t_all = self.get_t_all(nt=nt)
        n_cond = ev.shape[0]

        p_ch_cond_td = self.dtb(ev=ev)[0].permute([2, 0, 1])

        p_ch_cond_rt = []
        p_tnds = []
        for ch, (p_cond_td, tnd) in enumerate(zip(p_ch_cond_td, self.tnds)):
            p_tnd = tnd.get_p_tnd(t_all)
            p_cond_rt = npt.conv_t(
                p_cond_td[None],  # [1, cond, fr]
                p_tnd[None, None, :].expand([n_cond, 1, nt]),  # [cond, 1, fr]
                groups=n_cond
            )

            # CHECKED
            # plt.plot(*npys(t_all, p_cond_td[0, :]))
            # plt.plot(*npys(t_all, p_cond_rt[0, 0, :]));
            # plt.show()

            p_tnds.append(p_tnd)
            p_ch_cond_rt.append(p_cond_rt)

        p_ch_cond_rt = torch.stack(p_ch_cond_rt)
        p_cond_rt_ch = p_ch_cond_rt.permute([1, 2, 0])

        p_cond_rt_ch = self.lapse(p_cond_rt_ch)

        # CHECKING: to_normalize_within_cond will have to be True - should
        #  make RT prediction lines from serial parallel each other - if not
        #  fix something
        if self.to_normalize_within_cond:
            p_cond_rt_ch = p_cond_rt_ch / p_cond_rt_ch.sum([1, 2], keepdim=True)
            p_cond_rt_ch[torch.isnan(p_cond_rt_ch)] = 0.

        return p_cond_rt_ch

    def load_state_dict(self, state_dict, *args, strict=False, **kwargs):
        state_dict = sim1d.Dtb1D.update_state_dict(state_dict)
        super().load_state_dict(state_dict, *args, strict=strict, **kwargs)


class RTNonparam2D(Dtb2DRT):
    """
    Fit 1D RT distrib per 1D condition; take choice from data

    See also:
         Decision2D_bef_Cosyne2019/+Fit/+D2/+RT/+Td2Tnd/main_td2tnd.m
         Decision2D_bef_Cosyne2019/+Fit/+D2/+RT/+Td2Tnd/Main.m
    """
    kind = 'none'
    n_dim = 2

    def __init__(
            self,
            ev_cond_dim: np.ndarray,
            # n_cond_ch: np.ndarray,
            n_cond_rt_ch: np.ndarray,
            loc0=0.4,
            disper0=0.1,
            disper_ub=0.95,
            thres_n_tr=10,
            correct_only=False,
            exclude_0coh=False,
            sumto1_wi_cond=True,
            distrib_kind='lognorm',
            disper_kind='cv',
            **kwargs
    ):
        """

        :param ev_cond_dim: [condFlat, dim] = ev_cond_dim
        :param n_cond_ch: [condFlat, chFlat] = N(chFlat | condFlat)
        """
        super().__init__(**kwargs)

        n_cond_ch = npy(n_cond_rt_ch).sum(1)

        self.distrib_kind = distrib_kind
        assert disper_kind in ('cv', 'sd')
        self.disper_kind = disper_kind

        self.ev_cond_dim = ev_cond_dim
        self.nCondFlat = self.ev_cond_dim.shape[0]

        self.thres_n_tr = thres_n_tr
        self.sumto1_wi_cond = sumto1_wi_cond
        self.correct_only = correct_only
        self.exclude_0coh = exclude_0coh

        self.n_cond_ch = n_cond_ch
        self.p_cond__ch = torch.tensor(np2.sumto1(self.n_cond_ch, -1))

        self.conds_dim = []
        self.dCond_dim = []
        self.nCond_dim = []

        for dim in range(self.n_dim):
            conds, dcond = np.unique(self.ev_cond_dim[:, dim],
                                     return_inverse=True)
            self.conds_dim.append(conds)
            self.dCond_dim.append(dcond)
            self.nCond_dim.append(len(conds))
        self.dCond_dim = np.stack(self.dCond_dim, -1)

        if self.disper_kind == 'cv':
            self.loc_dim_cond = nn.ModuleList([
                ykt.BoundedParameter(
                    np.zeros([nCond, consts.N_CH]) + loc0, 0.05, 3.0
                ) for nCond in self.nCond_dim])
            self.disper_dim_cond = nn.ModuleList([
                ykt.BoundedParameter(
                    np.zeros([nCond, consts.N_CH]) + disper0, 0.01, disper_ub
                ) for nCond in self.nCond_dim])
        elif self.disper_kind == 'sd':
            # Determine mean and SD of RT in the data
            n_cond_rt_ch = npt.tensor(n_cond_rt_ch)
            n_cond0_cond1_rt_ch0_ch1 = n_cond_rt_ch.reshape([
                *self.nCond_dim, -1, 2, 2
            ])
            n_cond0_cond1_ch0_ch1 = n_cond0_cond1_rt_ch0_ch1.sum(2)
            # n_cond0_cond1_rt_ch0_ch1[
            #     n_cond0_cond1_ch0_ch1[:, :, None, :, :].expand_as(
            #         n_cond0_cond1_rt_ch0_ch1
            #     ) < self.thres_n_tr] = 1. / self.nt

            meanRt_cond0_cond1_ch0_ch1 = npy(npt.nan2v(npt.mean_distrib(
                npt.sumto1(n_cond0_cond1_rt_ch0_ch1, 2),
                self.t_all[None, None, :, None, None],
                axis=2
            ), 0.051).clamp_min(0.051))
            sdRt_cond0_cond1_ch0_ch1 = npy(npt.nan2v(npt.std_distrib(
                npt.sumto1(n_cond0_cond1_rt_ch0_ch1, 2),
                self.t_all[None, None, :, None, None],
                axis=2
            ), 0.051).clamp_min(0.051))

            to_excl = n_cond0_cond1_ch0_ch1 < self.thres_n_tr
            meanRt_cond0_cond1_ch0_ch1[to_excl] = np.nan
            sdRt_cond0_cond1_ch0_ch1[to_excl] = np.nan

            minMeanRt_dim_condDim_chDim = [
                np2.nan2v(np.nanmin(meanRt_cond0_cond1_ch0_ch1, (1, 3)),
                          0.051),
                np2.nan2v(np.nanmin(meanRt_cond0_cond1_ch0_ch1, (0, 2)),
                          0.051)
            ]
            minSdRt_dim_condDim_chDim = [
                np2.nan2v(np.nanmin(sdRt_cond0_cond1_ch0_ch1, (1, 3)),
                          0.051),
                np2.nan2v(np.nanmin(sdRt_cond0_cond1_ch0_ch1, (0, 2)),
                          0.051)
            ]
            self.loc_dim_cond = nn.ModuleList([
                ykt.BoundedParameter(
                    v / 2, 0.05, v * disper_ub
                ) for v in minMeanRt_dim_condDim_chDim])
            self.disper_dim_cond = nn.ModuleList([
                ykt.BoundedParameter(
                    np.sqrt((v ** 2) / 2), 0.02, np.sqrt((v ** 2) * disper_ub)
                ) for v in minSdRt_dim_condDim_chDim])

            # n_dim_condDim_rt_chDim = [
            #     n_cond0_cond1_rt_ch0_ch1.sum((1, 4)),
            #     n_cond0_cond1_rt_ch0_ch1.sum((0, 3))
            # ]
            # meanRt_dim_condDim_chDim = [
            #     npt.nan2v(
            #         npt.mean_distrib(
            #             npt.sumto1(v, 1),
            #             self.t_all[None, :, None], axis=1),
            #         0.051
            #     ).clamp_min(0.051) for v in n_dim_condDim_rt_chDim
            # ]
            # sdRt_dim_condDim_chDim = [
            #     npt.nan2v(
            #         npt.std_distrib(
            #             npt.sumto1(v, 1),
            #             self.t_all[None, :, None], axis=1),
            #         0.051
            #     ).clamp_min(0.051) for v in n_dim_condDim_rt_chDim
            # ]
            # self.loc_dim_cond = nn.ModuleList([
            #     ykt.BoundedParameter(
            #         v, 0.05, v * 2
            #     ) for v in meanRt_dim_condDim_chDim])
            # self.disper_dim_cond = nn.ModuleList([
            #     ykt.BoundedParameter(
            #         v, 0.05, ((v ** 2) * 2).sqrt()
            #     ) for v in sdRt_dim_condDim_chDim])

            # print([v[:] for v in self.disper_dim_cond])
            # print('--')


class RTNonparam2DSer(RTNonparam2D):
    kind = 'ser_np'

    def get_p_cond_td_ch(self, p_dim_cond_td_ch: torch.Tensor) -> torch.Tensor:
        return Dtb2DRTSerial.get_p_cond_td_ch(p_dim_cond_td_ch)

    def forward(self, ev: torch.Tensor, **kwargs
                ) -> Tuple[torch.Tensor, Any]:
        """
        Only uses ev of the first frame.
        :param ev: [condition, frame, dim, (meanvar)]
        :return: p_cond__rt_ch[condition, frame, ch_flat], None
        """
        nt = ev.shape[1]
        ev = ev[:, 0, :, 0]  # ev[cond, dim]
        n_tr = ev.shape[0]
        dCond_dim = np.empty([n_tr, self.n_dim], dtype=np.long)
        for dim in range(self.n_dim):
            for icond in range(self.nCond_dim[dim]):
                incl = ev[:, dim] == self.ev_cond_dim[icond, dim]
                dCond_dim[incl, dim] = icond

        t_all = self.get_t_all(nt=nt)

        # p_dim_cond_td_ch[dim, tr, td, ch]
        p_dim_cond_td_ch = torch.zeros(
            self.n_dim, self.nCondFlat, nt, consts.N_CH)
        for dim, (loc, disper) in enumerate(zip(
                self.loc_dim_cond, self.disper_dim_cond)):

            if self.disper_kind == 'cv':
                std = loc[:] * disper[:]
            elif self.disper_kind == 'sd':
                std = disper[:]
            else:
                raise ValueError()

            if self.distrib_kind == 'lognorm':
                p_cond_td_ch1 = npt.lognorm_pmf(
                        t_all[:, None, None],
                        loc[None, :, :], std[None, :, ]).transpose(0, 1)
            elif self.distrib_kind == 'invnorm':
                p_cond_td_ch1 = npt.inv_gaussian_pmf_mean_stdev(
                        t_all[:, None, None],
                        loc[None, :, :], std[None, :, ]).transpose(0, 1)
            else:
                raise ValueError()
            # p_cond_td_ch = npt.inv_gaussian_pmf_mean_stdev(
            #         t_all[None, :, None],
            #         loc[:, None, :], std[:, None, :])

            for iCond, cond in enumerate(self.conds_dim[dim]):
                incl = self.ev_cond_dim[:, dim] == cond
                p_dim_cond_td_ch[dim, incl, :, :] = p_cond_td_ch1[[iCond], :]

        p_cond_td_ch = self.get_p_cond_td_ch(p_dim_cond_td_ch)

        if self.sumto1_wi_cond:
            p_cond_td_ch = (
                    npt.sumto1(p_cond_td_ch, 1)
                    * self.p_cond__ch[:, None, :])
        else:
            p_cond_td_ch = p_cond_td_ch * self.p_cond__ch[:, None, :]

        p_cond_ch_td = p_cond_td_ch.permute([0, 2, 1])
        # incl[condFlat, chFlat]
        incl = self.n_cond_ch >= self.thres_n_tr

        if self.exclude_0coh:
            incl = incl & ((self.ev_cond_dim != 0).all(-1)[:, None])

        if self.correct_only:
            ch_sign = consts.ch_bool2sign(consts.CHS_ARRAY)

            for dim in range(consts.N_DIM):
                incl = incl & (
                    np.sign(self.ev_cond_dim[:, [dim]]) != -ch_sign[[dim], :])

        # axs = plt2.GridAxes(1, 4)
        # for col in range(4):
        #     plt.sca(axs[0, col])
        #     plt.imshow(incl[:, col].reshape([9, 9]))
        # plt.show()  # CHECKED

        p_cond_ch_td[~incl] = self.p_cond__ch[~incl][:, None] / nt
        p_cond_td_ch = p_cond_ch_td.permute([0, 2, 1])

        # p_cond_td_ch = p_cond_td_ch * torch.tensor(
        #     incl[:, None, :], dtype=torch.float)

        # for cond in range(self.nCondFlat):
        #     for ch in range(consts.N_CH_FLAT):
        #         p_cond__ch

        p_tr__td_ch = torch.empty(n_tr, nt, consts.N_CH_FLAT)
        for iCond, ev_cond1 in enumerate(self.ev_cond_dim):
            incl = torch.all(ev == torch.tensor(ev_cond1[None, :]), dim=1)
            p_tr__td_ch[incl, :] = p_cond_td_ch[iCond, :]

        # if not self.sumto1_wi_cond:
        #     print(p_tr__td_ch.sum([1, 2]))  # CHECKED sum within cond

        return p_tr__td_ch, None


class RTNonparam2DPar(RTNonparam2DSer):
    kind = 'par_np'

    def get_p_cond_td_ch(self, p_dim_cond_td_ch: torch.Tensor) -> torch.Tensor:
        return Dtb2DRTParallel.get_p_cond_td_ch(p_dim_cond_td_ch)


def ____Plot____():
    pass


def plot_p_ch_vs_ev(ev_cond: Union[torch.Tensor, np.ndarray],
                    n_ch: Union[torch.Tensor, np.ndarray],
                    style='pred',
                    ax: plt.Axes = None,
                    dim_rel=0,
                    group_dcond_irr : Iterable[Iterable[int]] = None,
                    cmap: Union[str, Callable] = 'cool',
                    kw_plot=(),
                    ) -> Iterable[plt.Line2D]:
    """
    @param ev_cond: [condition, dim] or [condition, frame, dim, (mean, var)]
    @type ev_cond: torch.Tensor
    @param n_ch: [condition, ch] or [condition, rt_frame, ch]
    @type n_ch: torch.Tensor
    @return: hs[cond_irr][0] = Line2D, conds_irr
    """
    if ax is None:
        ax = plt.gca()
    if ev_cond.ndim != 2:
        assert ev_cond.ndim == 4
        ev_cond = npt.p2st(ev_cond.mean(1))[0]
    if n_ch.ndim != 2:
        assert n_ch.ndim == 3
        n_ch = n_ch.sum(1)

    ev_cond = npy(ev_cond)
    n_ch = npy(n_ch)
    n_cond_all = n_ch.shape[0]
    ch_rel = np.repeat(
        np.array(consts.CHS[dim_rel])[None, :], n_cond_all, 0)
    n_ch = n_ch.reshape([-1])
    ch_rel = ch_rel.reshape([-1])

    dim_irr = consts.get_odim(dim_rel)
    conds_rel, dcond_rel = np.unique(ev_cond[:, dim_rel], return_inverse=True)
    conds_irr, dcond_irr = np.unique(
        np.abs(ev_cond[:, dim_irr]), return_inverse=True)

    if group_dcond_irr is not None:
        conds_irr, dcond_irr = group_conds(conds_irr, dcond_irr,
                                           group_dcond_irr)

    n_conds = [len(conds_rel), len(conds_irr)]

    n_ch_rel = npg.aggregate(
        np.stack([
            ch_rel,
            np.repeat(dcond_irr[:, None], consts.N_CH_FLAT, 1).flatten(),
            np.repeat(dcond_rel[:, None], consts.N_CH_FLAT, 1).flatten(),
        ]),
        n_ch, 'sum', [consts.N_CH, n_conds[1], n_conds[0]]
    )
    p_ch_rel = n_ch_rel[1] / n_ch_rel.sum(0)

    hs = []
    for dcond_irr1, p_ch1 in enumerate(p_ch_rel):
        if type(cmap) is str:
            color = plt.get_cmap(cmap, n_conds[1])(dcond_irr1)
        else:
            color = cmap(n_conds[1])(dcond_irr1)
        kw1 = get_kw_plot(style, color=color, **dict(kw_plot))
        h = ax.plot(conds_rel, p_ch1, **kw1)
        hs.append(h)
    plt2.box_off(ax=ax)
    x_lim = ax.get_xlim()
    plt2.detach_axis('x', amin=x_lim[0], amax=x_lim[1], ax=ax)
    plt2.detach_axis('y', amin=0, amax=1, ax=ax)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(['0', '', '1'])
    ax.set_xlabel('evidence')
    ax.set_ylabel(r"$\mathrm{P}(z=1 \mid c)$")

    return hs, conds_irr


def group_conds(conds_irr, dcond_irr, group_dcond_irr):
    dcond_irr1 = np.empty_like(dcond_irr)
    conds_irr1 = []
    for group, dconds_in_group in enumerate(group_dcond_irr):
        incl = np.any(dcond_irr[:, None]
                      == np.array(dconds_in_group)[None, :], 1)
        dcond_irr1[incl] = group
        conds_irr1.append(conds_irr[dconds_in_group])
    dcond_irr = dcond_irr1
    conds_irr = conds_irr1
    return conds_irr, dcond_irr


def plot_rt_vs_ev(
        ev_cond: Union[torch.Tensor, np.ndarray],
        n_cond__rt_ch: Union[torch.Tensor, np.ndarray],
        dim_rel=0,
        group_dcond_irr: Iterable[Iterable[int]] = None,
        correct_only=True,
        cmap: Union[str, Callable] = 'cool',
        kw_plot=(),
        **kwargs
) -> Tuple[Sequence[Sequence[plt.Line2D]], Sequence[Sequence[float]]]:
    """
    @param ev_cond: [condition]
    @type ev_cond: torch.Tensor
    @param n_cond__rt_ch: [condition, frame, ch]
    @type n_cond__rt_ch: torch.Tensor
    @return:
    """
    if ev_cond.ndim != 2:
        assert ev_cond.ndim == 4
        ev_cond = npt.p2st(ev_cond.mean(1))[0]
    assert n_cond__rt_ch.ndim == 3

    ev_cond = npy(ev_cond)
    n_cond__rt_ch = npy(n_cond__rt_ch)
    nt = n_cond__rt_ch.shape[1]
    n_ch = n_cond__rt_ch.shape[2]
    n_cond_all = n_cond__rt_ch.shape[0]

    ix_sort = np.argsort(ev_cond[:, dim_rel])
    ev_cond = ev_cond[ix_sort]
    n_cond__rt_ch = n_cond__rt_ch[ix_sort]

    dim_irr = consts.get_odim(dim_rel)
    conds_rel, dcond_rel = np.unique(ev_cond[:, dim_rel], return_inverse=True)
    conds_irr, dcond_irr = np.unique(
        np.abs(ev_cond[:, dim_irr]), return_inverse=True)

    if group_dcond_irr is not None:
        conds_irr, dcond_irr = group_conds(conds_irr, dcond_irr,
                                           group_dcond_irr)

    n_conds = [len(conds_rel), len(conds_irr)]

    if correct_only:  # simply split at zero
        hs = []
        rts = []
        for dcond_irr1 in range(n_conds[1]):
            for ch1 in range(consts.N_CH):
                ch_sign = consts.ch_bool2sign(ch1)

                incl = dcond_irr == dcond_irr1
                ev_cond1 = conds_rel

                n_cond__rt_ch1 = np.empty([n_conds[0], nt, n_ch])
                for dcond_rel1 in range(n_conds[0]):
                    incl1 = incl & (dcond_rel1 == dcond_rel)
                    n_cond__rt_ch1[dcond_rel1] = n_cond__rt_ch[incl1].sum(0)

                if type(cmap) is str:
                    color = plt.get_cmap(cmap, n_conds[1])(dcond_irr1)
                else:
                    color = cmap(n_conds[1])(dcond_irr1)

                chs = np.array(consts.CHS)
                n_cond__rt_ch11 = np.zeros(n_cond__rt_ch1.shape[:2]
                                           + (consts.N_CH,))

                # # -- Pool across ch_irr
                for ch_rel in range(consts.N_CH):
                    incl = chs[dim_rel] == ch_rel
                    n_cond__rt_ch11[:, :, ch_rel] = n_cond__rt_ch1[
                                                    :, :, incl].sum(-1)

                hs1, rts1 = sim1d.plot_rt_vs_ev(
                    ev_cond1, n_cond__rt_ch11,
                    color=color,
                    correct_only=correct_only,
                    kw_plot=kw_plot,
                    **kwargs,
                )
                hs.append(hs1)
                rts.append(rts1)
    else:
        raise NotImplementedError()
    return hs, rts


def plot_rt_distrib(
        n_cond_rt_ch: np.ndarray, ev_cond_dim: np.ndarray,
        abs_cond=True,
        lump_wrong=True,
        dt=consts.DT,
        colors=None, alpha=1.,
        alpha_face=0.5,
        smooth_sigma_sec=0.05,
        to_normalize_max=False,
        to_cumsum=False,
        to_exclude_last_frame=True,
        to_skip_zero_trials=False,
        label='',
        # to_exclude_bins_wo_trials=10,
        kw_plot=(),
        fig=None,
        axs=None,
        to_use_sameaxes=True,
):
    """

    :param n_cond_rt_ch:
    :param ev_cond_dim:
    :param abs_cond:
    :param lump_wrong:
    :param dt:
    :param gs:
    :param colors:
    :param alpha:
    :param smooth_sigma_sec:
    :param kw_plot:
    :param axs:
    :return: axs, p_cond01__rt_ch01, p_cond01__rt_ch01_sm, hs
    """
    if colors is None:
        colors = ['red', 'blue']
    elif type(colors) is str:
        colors = [colors] * 2
    else:
        assert len(colors) == 2

    nt = n_cond_rt_ch.shape[1]
    t_all = np.arange(nt) * dt + dt

    out = np.meshgrid(
        np.unique(ev_cond_dim[:, 0]),
        np.unique(ev_cond_dim[:, 1]),
        np.arange(nt), np.arange(2), np.arange(2),
        indexing='ij'
    )
    cond0, cond1, fr, ch0, ch1 = [v.flatten() for v in out]

    from copy import deepcopy
    n0 = deepcopy(n_cond_rt_ch)
    if to_exclude_last_frame:
        n0[:, -1, :] = 0.
    n0 = n0.flatten()

    def sign_cond(v):
        v1 = np.sign(v)
        v1[v == 0] = 1
        return v1

    if abs_cond:
        ch0 = consts.ch_bool2sign(ch0)
        ch1 = consts.ch_bool2sign(ch1)

        # 1 = correct, -1 = wrong
        ch0 = sign_cond(cond0) * ch0
        ch1 = sign_cond(cond1) * ch1

        cond0 = np.abs(cond0)
        cond1 = np.abs(cond1)

        ch0 = consts.ch_sign2bool(ch0).astype(np.int)
        ch1 = consts.ch_sign2bool(ch1).astype(np.int)
    else:
        raise ValueError()

    if lump_wrong:
        # treat all choices as correct when cond == 0
        ch00 = ch0 | (cond0 == 0)
        ch10 = ch1 | (cond1 == 0)

        ch0 = (ch00 & ch10)
        ch1 = np.ones_like(ch00, dtype=np.int)

    cond_dim = np.stack([cond0, cond1], -1)

    conds = []
    dcond_dim = []
    for cond in cond_dim.T:
        conds1, dcond1 = np.unique(cond, return_inverse=True)
        conds.append(conds1)
        dcond_dim.append(dcond1)
    dcond_dim = np.stack(dcond_dim)

    n_cond01_rt_ch01 = npg.aggregate([
        *dcond_dim, fr, ch0, ch1
    ], n0, 'sum', [*(np.amax(dcond_dim, 1) + 1), nt, consts.N_CH, consts.N_CH])

    p_cond01__rt_ch01 = np2.nan2v(n_cond01_rt_ch01
                                  / n_cond01_rt_ch01.sum((2, 3, 4),
                                                         keepdims=True))

    n_conds = p_cond01__rt_ch01.shape[:2]

    # # CHECKED
    # print(p_cond01__rt_ch01.sum((2, 3))[:, :, 1])

    if axs is None:
        axs = plt2.GridAxes(
            n_conds[1], n_conds[0],
            left=0.6, right=0.3,
            bottom=0.45, top=0.74,
            widths=[1], heights=[1],
            wspace=0.04, hspace=0.04,
        )
        # if fig is None:
        #     fig = plt.figure(figsize=[6, 6])
        # gs = plt.GridSpec(
        #     nrows=n_conds[1], ncols=n_conds[0],
        #     left=0.1, right=0.95,
        #     bottom=0.075, top=0.9,
        #     figure=fig,
        # )
        # # fig = plt.figure(figsize=[7.5, 7.5])
        # # gs = plt.GridSpec(
        # #     nrows=n_conds[1], ncols=n_conds[0],
        # #     left=0.05, right=0.98,
        # #     bottom=0.07, top=0.95,
        # #     hspace=0.04
        # # )
        #
        # axs = np.empty(n_conds[1::-1], dtype=np.object)
        # for row in range(axs.shape[0]):
        #     for col in range(axs.shape[1]):
        #         axs[row, col] = plt.subplot(gs[row, col])

    kw_label = {
        'fontsize': 12,
    }
    pad = 8
    axs[0, 0].set_title('strong\nmotion', pad=pad, **kw_label)
    axs[0, -1].set_title('weak\nmotion', pad=pad, **kw_label)
    axs[0, 0].set_ylabel('strong\ncolor', labelpad=pad, **kw_label)
    axs[-1, 0].set_ylabel('weak\ncolor', labelpad=pad, **kw_label)

    if smooth_sigma_sec > 0:
        from scipy import signal, stats
        sigma_fr = smooth_sigma_sec / dt
        width = np.ceil(sigma_fr * 2.5).astype(np.int)
        kernel = stats.norm.pdf(np.arange(-width, width+1), 0, sigma_fr)
        kernel = np2.vec_on(kernel, 2, 5)
        p_cond01__rt_ch01_sm = signal.convolve(
            p_cond01__rt_ch01, kernel,
            mode='same'
        )
    else:
        p_cond01__rt_ch01_sm = p_cond01__rt_ch01.copy()

    if to_cumsum:
        p_cond01__rt_ch01_sm = np.cumsum(p_cond01__rt_ch01_sm, axis=2)

    if to_normalize_max:
        p_cond01__rt_ch01_sm = np2.nan2v(
            p_cond01__rt_ch01_sm
            / np.amax(np.abs(p_cond01__rt_ch01_sm),
                      (2, 3, 4), keepdims=True)
        )

    n_row = n_conds[1]
    n_col = n_conds[0]
    for dcond0 in range(n_conds[0]):
        for dcond1 in range(n_conds[1]):
            row = n_row - 1 - dcond1
            col = n_col - 1 - dcond0

            ax = axs[row, col]  # type: plt.Axes

            for ch0 in [0, 1]:
                for ch1 in [0, 1]:
                    if lump_wrong and ch1 == 0:
                        continue

                    p1 = p_cond01__rt_ch01_sm[dcond0, dcond1, :, ch0, ch1]

                    kw = {
                        'linewidth': 1,
                        'color': colors[ch1],
                        'alpha': alpha,
                        'zorder': 1,
                        **dict(kw_plot)
                    }

                    y = p1 * consts.ch_bool2sign(ch0)

                    p_cond01__rt_ch01_sm[dcond0, dcond1, :, ch0, ch1] = y

                    # if 'linestyle' in kw_plot.keys():  # CHECKED
                    #     if kw_plot['linestyle'] == ':':
                    #         print(np.sum(np.abs(y)))
                    #         print('--')

                    if to_skip_zero_trials and np.sum(np.abs(y) ) < 1e-2:
                        h = None
                    else:
                        h = ax.plot(
                            t_all, y,
                            label=label if ch0 == 1 and ch1 == 1 else None,
                            **kw)
                        ax.fill_between(
                            t_all, 0, y,
                            ec='None',
                            fc=kw['color'],
                            alpha=alpha_face,
                            zorder=-1
                        )
                    plt2.box_off(ax=ax)

            ax.axhline(0, color='k', linewidth=0.5)
            ax.set_yticklabels([])
            if row < n_row - 1 or col > 0:
                ax.set_xticklabels([])
                ax.set_xticks([])
                plt2.box_off(['bottom'], ax=ax)
            else:
                ax.set_xlabel('RT (s)')
            # if col > 0:
            ax.set_yticks([])
            ax.set_yticklabels([])
            plt2.box_off(['left'], ax=ax)

            plt2.detach_axis('x', 0, 5, ax=ax)
    if to_use_sameaxes:
        plt2.sameaxes(axs)
    axs[-1, 0].set_xlabel('RT (s)')

    return axs, p_cond01__rt_ch01, p_cond01__rt_ch01_sm, h


def plot_p_tnd1(model, d=None, data_mode=None):
    fig = plt.figure('p_tnd', figsize=[4, 3.5])
    gs = plt.GridSpec(
        nrows=2, ncols=2,
        left=0.2, right=0.95, bottom=0.25, top=0.95,
    )
    for ch0 in range(consts.N_CH):
        for ch1 in range(consts.N_CH):
            ch_flat = consts.ch_by_dim2ch_flat(np.array([ch0, ch1]))
            ax = plt.subplot(gs[ch1, ch0])  # type: plt.Axes # noqa
            model.tnds[ch_flat].plot_p_tnd()
            ax.set_ylim(top=1)
            ax.set_yticks([0, 1])
            if ch1 == 0:
                ax.set_xticklabels([])
                ax.set_xlabel('')
            if ch0 == 1:
                ax.set_yticklabels([])
                ax.set_ylabel('')
    return fig, d


def plot_bound(model, d=None, data_mode=None):
    fig = plt.figure('bound', figsize=[4, 2])
    gs = plt.GridSpec(
        nrows=1, ncols=2,
        left=0.2, right=0.95, bottom=0.25, top=0.95
    )
    for dim_rel in range(consts.N_DIM):
        plt.subplot(gs[0, dim_rel])
        if hasattr(model.dtb, 'dtb1ds'):
            model.dtb.dtb1ds[dim_rel].plot_bound(color='k')
    return fig, d


def plot_ch(model, d, data_mode='train_valid'):
    data = d['data_' + data_mode]
    out = d['out_' + data_mode]
    target = d['target_' + data_mode]

    fig = plt.figure('ch', figsize=[4, 2])
    gs = plt.GridSpec(
        nrows=1, ncols=2,
        left=0.2, right=0.95, bottom=0.25, top=0.95
    )
    for dim_rel in range(consts.N_DIM):
        plt.subplot(gs[0, dim_rel])
        plot_p_ch_vs_ev(data, out, style='pred',
                        dim_rel=dim_rel)
        plot_p_ch_vs_ev(data, target, style='data',
                        dim_rel=dim_rel)
    return fig, d


def plot_rt(model, d, data_mode='train_valid'):
    data = d['data_' + data_mode]
    out = d['out_' + data_mode]
    target = d['target_' + data_mode]

    fig = plt.figure('rt', figsize=[4, 2])
    gs = plt.GridSpec(
        nrows=1, ncols=2,
        left=0.2, right=0.95, bottom=0.25, top=0.95
    )
    for dim_rel in range(consts.N_DIM):
        plt.subplot(gs[0, dim_rel])
        plot_rt_vs_ev(data, out, style='pred',
                      dim_rel=dim_rel, dt=model.dt)
        plot_rt_vs_ev(data, target, style='data',
                      dim_rel=dim_rel, dt=model.dt)
    return fig, d


def plot_rt_distrib1(model, d, data_mode='train_valid'):
    data = d['data_' + data_mode]
    out = d['out_' + data_mode]
    target = d['target_' + data_mode]

    fig = plt.figure('rtdstr', figsize=[4, 4])

    ev_cond_dim = npy(data[:, :, :, 0].sum(1))
    axs = plot_rt_distrib(
        npy(out),
        ev_cond_dim,
        alpha_face=0.,
        colors=['b', 'b'],
        fig=fig
    )[0]
    axs = plot_rt_distrib(
        npy(target),
        ev_cond_dim,
        alpha_face=0.,
        colors=['k', 'k'],
        axs=axs,
    )[0]
    return fig, d


def plot_params(model, d=None, data_mode=None):
    fig = plt.figure('params', figsize=(6, 12))
    gs = plt.GridSpec(nrows=1, ncols=1, left=0.35)
    ax = plt.subplot(gs[0, 0])
    model.plot_params(ax=ax)
    return fig, d


def ____Fit___():
    pass


def fun_data(data: Data2DRT, mode='all', i_fold_valid=0, epoch=0,
             n_fold_valid=1, i_fold_test=0, n_fold_test=1,
             mode_train='all', to_debug=False):
    ev_cond_fr_dim_meanvar, n_cond_rt_ch = data.get_data_by_cond(
        mode=mode, i_fold_valid=i_fold_valid, epoch=epoch,
        n_fold_valid=n_fold_valid,
        i_fold_test=i_fold_test, n_fold_test=n_fold_test,
        mode_train=mode_train, to_debug=to_debug
    )[:2]
    return ev_cond_fr_dim_meanvar, n_cond_rt_ch


def fun_loss(
        p_cond__rt_ch_pred: torch.Tensor,
        n_cond__rt_ch_data: torch.Tensor,
        ignore_hard_RT=False,
        conds: Union[torch.Tensor, np.ndarray] = None,
        **kwargs
) -> torch.Tensor:
    """
    :param conds: [cond, dim]
    """
    if ignore_hard_RT:
        conds = npy(conds)
        ix_conds_to_ignore_rt = np.any(
            conds == np.amax(np.abs(conds), axis=0),
            axis=1
        )
    else:
        ix_conds_to_ignore_rt = None
    return sim1d.fun_loss(p_cond__rt_ch_pred,
                          n_cond__rt_ch_data,
                          ix_conds_to_ignore_rt=ix_conds_to_ignore_rt,
                          **kwargs)


def fit_dtb(model: FitRT2D,
            data: Data2DRT,
            n_fold_valid=1,
            n_fold_test=1,
            i_fold_test=0,
            mode_train='all',
            ignore_hard_RT=False,
            to_debug=False,
            **kwargs
            ) -> (float, dict, dict, ykt.PlotFunsType):
    """
    Provide functions fun_data() and plot_*() to ykt.optimize().
    See ykt.optimize() for details about fun_data and plot_*
    :param model:
    :param data:
    :param n_fold_valid
    :param mode_train: 'all'|'easiest' - which conditions to use in training
    :param kwargs: fed to ykt.optimize()
    :return: best_loss, best_state
    """

    def fun_data1(mode='all', i_fold_valid=0, epoch=0, n_fold_valid=1):
        ev_cond_fr_dim_meanvar, n_cond_rt_ch = fun_data(
            data, mode=mode, i_fold_valid=i_fold_valid, epoch=epoch,
            n_fold_valid=n_fold_valid,
            i_fold_test=i_fold_test,
            n_fold_test=n_fold_test,
            mode_train=mode_train, to_debug=to_debug
        )
        return ev_cond_fr_dim_meanvar, n_cond_rt_ch

    def fun_loss1(
            p_cond__rt_ch_pred: torch.Tensor,
            n_cond__rt_ch_data: torch.Tensor,
            **kwargs
    ):
        return fun_loss(
            p_cond__rt_ch_pred,
            n_cond__rt_ch_data,
            ignore_hard_RT=ignore_hard_RT,
            conds=data.ev_cond_dim,
            **kwargs
        )

    kw_optim = {**{
        **{
            'n_fold_valid': n_fold_valid
        },
        **kwargs}, **{
        'optimizer_kind': 'Adam',
        'learning_rate': .1,
        'patience': 100,
        'reduce_lr_after': 25,
        'thres_patience': 1e-4,
        'to_print_grad': False,  # CHECKED
    }}
    plotfuns = [
            ('ch', plot_ch),
            ('rt', plot_rt),
            ('bound', plot_bound),
            ('tnd', plot_p_tnd1),
            # ('rtdstr', plot_rt_distrib1),
            ('params', plot_params)
        ]

    best_loss, best_state, d = ykt.optimize(
        model, fun_data1, fun_loss1,
        plotfuns=plotfuns,
        **kw_optim
    )[:3]

    with torch.no_grad():
        for data_mode in ['train_valid', 'test', 'all']:
            inp, target = fun_data1(data_mode)
            out = model(inp)

            for loss_kind in ['CE', 'NLL', 'BIC']:
                if loss_kind == 'CE':
                    loss = fun_loss1(out, target, to_average=True,
                                    base_n_bin=True)
                elif loss_kind in ['NLL', 'BIC']:
                    loss = fun_loss1(out, target, to_average=False,
                                    base_n_bin=False)
                if loss_kind == 'BIC':
                    n = npy(target.sum())
                    k = np.sum([
                        v.numel() if v.requires_grad else 0
                        for v in model.parameters()
                    ])
                    loss = loss * 2 + k * np.log(n)
                    d['loss_ndata_%s' % data_mode] = n
                    d['loss_nparam'] = k

                d['loss_%s_%s' % (loss_kind, data_mode)] = loss

    return best_loss, best_state, d, plotfuns


def ____Demo____():
    pass


def demo_interpolate():
    """
    Test 1D interpolation along the time axis.

    NOTE: F.interpolate is not suitable for my purpose since it's restricted to
      rational ratios (output shape should be int); use F.grid_sample instead.
      F.grid_sample is for 2D but I can just add one dimension for that.
    """
    src = torch.tensor([[[0., 1., 0., 0.]]])
    dst = F.interpolate(src, scale_factor=2.152, mode='linear')
    print(dst)
    print(dst.shape)
    print(np.array(dst.shape) / np.array(src.shape))


def ____Main____():
    pass


def main():
    subsample_factor=5

    ev0 = get_demo_ev()
    model = FitRT2D(dtb2d=Dtb2DRTParallel)

    # ev_cond subsampled by subsample_factor
    # stack mean_ev and var_ev
    ev = torch.stack(list(sim1d.subsample_ev(ev0, subsample_factor)), -1)

    t_st = time.time()
    p_cond__rt_ch0 = model(ev)
    t_el = time.time() - t_st
    print('Elapsed time: %g sec.' % t_el)
    # p_cond__rt_ch

    n_cond_rt_ch_dat = simulate_p_cond__rt_ch(p_cond__rt_ch0, n_sample=100)
    model.zero_grad()
    loss = fun_loss(p_cond__rt_ch0, n_cond_rt_ch_dat)
    loss.backward()

    print(loss)
    grads = odict([
        (k, v.grad) for (k, v) in model.named_parameters()
    ])
    pprint(grads)

    plt.subplot(2, 1, 1)
    plot_p_ch_vs_ev(ev, n_cond_rt_ch_dat, style='data')
    plot_p_ch_vs_ev(ev, p_cond__rt_ch0, style='pred')

    plt.subplot(2, 1, 2)
    plot_rt_vs_ev(ev, n_cond_rt_ch_dat, style='data')
    plot_rt_vs_ev(ev, p_cond__rt_ch0, style='pred')

    plt.show()
    print('--')


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_num_threads(6)
    torch.set_default_dtype(torch.double)

    main()