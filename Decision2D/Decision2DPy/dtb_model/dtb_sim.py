#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from copy import copy
from copy import deepcopy
from pprint import pprint
from typing import Union

import torch
from torch.distributions import Normal

from lib.pylabyk import yktorch as ykt
from lib.pylabyk import numpytorch as npt
from lib.pylabyk import np2, plt2
from lib.pylabyk.numpytorch import npy, npys

from data_2d import consts


class State2D(object):
    def __init__(self, ev=None):
        """
        dv[tr, dim]: decision variable
        unabsorbed[tr, dim]: whether the state remained unabsorbed
        dv_open[tr, dim]: which dimension is accumulating dv
        """
        self.n_repeat = 1
        # state_at_time: recorded per time
        self.key_state_at_time = {
            'dv', 'unabsorbed', 'dv_open',
            'log_p_ddv', 'log_p_unabs'
        }
        # state_const: recorded once
        self.key_state_const = {
            'n_dim', 'n_tr', 'nt', 'dt', 't',
            'ch', 'rt', 'p_ch_rt'
        }

        if ev is None:
            ev = np.empty([0, consts.N_DIM, consts.NT])
        self.n_tr = ev.shape[0]
        self.n_dim = ev.shape[1]
        self.nt = ev.shape[2]
        self.dt = consts.DT
        self.t = np.arange(self.nt, dtype=np.double) * self.dt

        self.dv = torch.zeros(self.n_tr, self.n_dim)
        self.unabsorbed = torch.zeros(self.n_tr, self.n_dim,
                                      dtype=torch.bool)
        self.dv_open = torch.ones(self.n_tr, self.n_dim, dtype=torch.bool)
        self.log_p_ddv = torch.zeros(self.n_tr, self.n_dim)

        self.n_ch = consts.N_CH
        self.ch = -torch.ones(self.n_tr, self.n_dim, dtype=torch.long)
        self.rt = -torch.ones(self.n_tr, self.n_dim)

        self.p_ch_rt = torch.zeros(self.n_tr, self.n_dim)
        self.log_p_unabs = torch.zeros(self.n_tr, self.n_dim)

    def get_dict_state_at_time(self, return_numpy=True):
        if return_numpy:
            return {k: npy(self.__dict__[k]) for k in self.key_state_at_time}
        else:
            return {k: self.__dict__[k].clone() for k in self.key_state_at_time}

    def get_dict_state_const(self, return_numpy=True):
        if return_numpy:
            return {k: npy(self.__dict__[k]) for k in self.key_state_const}
        else:
            return {k: self.__dict__[k].clone()
                    if torch.is_tensor(k)
                    else copy(self.__dict__[k])
                    for k in self.key_state_at_time
                    }

    @property
    def dict_numpy(self):
        d = self.get_dict_state_const(return_numpy=True)
        d.update(self.get_dict_state_at_time(return_numpy=True))
        return d

    def dict_torch(self):
        d = self.get_dict_state_const(return_numpy=False)
        d.update(self.get_dict_state_at_time(return_numpy=False))
        return d

    def summarize_across_time(self, list_state_at_time, return_numpy=True):
        state_at_times = np2.listdict2dictlist(list_state_at_time)
        for k in state_at_times.keys():
            self.__dict__[k] = \
                np.stack(state_at_times[k], axis=-1) if return_numpy \
                else torch.stack(state_at_times[k], -1)
        return self

    def update(self, dict):
        for k in dict.keys():
            self.__dict__[k] = dict[k]

    def update_with_data(self, dict):
        for k in dict.keys():
            self.__dict__[k] = dict[k]
        self.key_state_at_time.update(dict.keys())

    def filt(self, incl, to_copy=True):
        if to_copy:
            s = deepcopy(self)
        else:
            s = self
        s.update(np2.filt_dict(self.dict_numpy, incl))
        if to_copy:
            return s


def repeat_trials(v, n_repeat=1, n_tr=None):
    """
    Repeat each item on the first (trial) dimension
    @type v: Union[np.ndarray, torch.Tensor]
    @type n_repeat: int
    @type n_tr: int
    @rtype: dict
    """
    is_array = isinstance(v, np.ndarray)
    is_tensor = torch.is_tensor(v)
    if is_array or is_tensor:
        if n_tr is None:
            n_tr = v.shape[0]

        if v.shape[0] == n_tr * n_repeat:
            if is_array:
                return np.tile(v, reps=n_repeat)
            elif is_tensor:
                return v.repeat(v, [n_repeat] + [1] * (v.ndim - 1))
        else:
            return v
    else:
        return v


def reshape_to_particle_by_trial(v, n_repeat=1, n_tr=None):
    """
    Make the first dims [n_repeat * tr, ...] -> [n_repeat, tr, ...]
    @type v: Union[np.ndarray, torch.Tensor]
    @type n_repeat: int
    @type n_tr: int
    @rtype: Union[np.ndarray, torch.Tensor]
    """
    if isinstance(v, np.ndarray) or torch.is_tensor(v):
        if n_tr is None:
            n_tr = v.shape[0] // n_repeat

        if v.shape[0] == n_tr * n_repeat:
            return v.reshape(
                [n_repeat, n_tr] + list(v.shape[1:])
            )
        else:
            return v
    else:
        return v


class Dtb2DSim(ykt.BoundedModule):
    def __init__(
            self,
            bound0=(1., 1.),
            drift_per_ev0=(20., 20.),
            dt=None,
    ):
        super().__init__()
        self.key_model_settings = {
            'model_kind', 'n_dim', 'n_ch', 'dt'
        }

        self.model_kind = 'unset'
        self.state_class = State2D

        self.n_dim = consts.N_DIM
        self.n_ch = consts.N_CH
        if dt is None:
            dt = consts.DT
        self.dt = dt

        # To prevent "leaf variable has been moved into the graph
        # interior", need to restrict direct assignment to parameters to
        # their construction
        self.cev_t0 = ykt.BoundedParameter([0., 0.], -0.5, 0.5)
        self.log10_var_t0 = ykt.BoundedParameter([-6., -6.], -9., 0.)
        self.bound0 = ykt.BoundedParameter(bound0, 0.01, 3)
        self.drift_per_ev0 = ykt.BoundedParameter(drift_per_ev0, 0.01, 50)

        # self.register_bounded_parameter('cev_t0', [0., 0.], -0.5, 0.5)
        # self.register_bounded_parameter('log10_var_t0', [-6., -6.], -9., 0.)
        # self.register_bounded_parameter('bound0', [1., 1.], 0.01, 3)
        # self.register_bounded_parameter('drift_per_ev0', [20., 20.], 0.01, 50)

    def init_sim_state(self, ev):
        """
        @param ev: [trial, dim, time]
        @type ev: torch.Tensor
        @rtype: State2D
        """
        return self.state_class(ev)

    def get_dict_model_settings(self):
        return {
            k: self.__dict__[k] for k in self.key_model_settings
        }


class State2DPar(State2D):
    pass

class Dtb2DParSim(Dtb2DSim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_kind = 'parallel'
        self.state_class = State2DPar

    @property
    def sigma_t0(self):
        return torch.exp(
            self.log10_var_t0[:] * torch.log(torch.tensor(10.)) / 2
        )

    @property
    def sigma_per_dt(self):
        return torch.sqrt(torch.tensor(self.dt))

    def get_drift_per_ev(self, t):
        """
        @type t: double
        @return: drift_per_ev[dim]
        @rtype: torch.FloatTensor
        """
        return self.drift_per_ev0[:]

    def get_bound(self, t):
        """
        @type t: double
        @return: bound[dim, ch]
        @rtype: torch.FloatTensor
        """
        return torch.cat([
            -self.bound0[:, None],
            self.bound0[:, None]
            ], 1)

    def init_sim_state(self, ev):
        """
        @rtype: State2DPar
        """
        state = super().init_sim_state(ev) #type: State2DPar

        state.dv, distrib = npt.normrnd(
            self.cev_t0[:], self.sigma_t0[:], (state.n_tr,),
            return_distrib=True
        )
        state.log_p_unabs = distrib.log_prob(state.dv)

        state.unabsorbed = torch.ones(
            state.n_tr, state.n_dim, dtype=torch.bool
        )
        state.dv_open = torch.ones(
            state.n_tr, state.n_dim, dtype=torch.bool
        )
        state.log_p_ddv = torch.zeros_like(state.dv)
        return state

    def get_ddv(self, ev_t, t):
        """
        @param ev_t: [trial, dim]
        @type ev_t: torch.Tensor
        @type t: float
        @return: (ddv, stimulus_ended, log_p_ddv)

        ddv[trial, dim] = drift + diffusion. 0 if stimulus_ended.

        stimulus_ended[trial, dim]

        log_p_ddv[trial, dim]

        @rtype: (torch.FloatTensor, torch.BoolTensor,
        torch.FloatTensor)
        """
        drift = self.get_drift_per_ev(t) * ev_t
        stimulus_ended = torch.isnan(ev_t)

        ddv, distrib = npt.normrnd(drift * self.dt, self.sigma_per_dt,
                                   return_distrib=True)
        log_p_ddv = distrib.log_prob(ddv)

        # ddv = ddv * (~stimulus_ended).double()
        # log_p_ddv = log_p_ddv * (~stimulus_ended).double()

        ddv[stimulus_ended] = 0.
        log_p_ddv[stimulus_ended] = 0.

        return ddv, stimulus_ended, log_p_ddv

    def force_absorb(self, ev_t, t, s):
        """
        @param ev_t:
        @param t:
        @type s: State2DPar
        @return:
        """

        # Force absorb if unabsorbed when the stimulus is terminated
        if t == (s.nt - 1) * self.dt:
            to_force_absorb = s.unabsorbed
        else:
            to_force_absorb = torch.isnan(ev_t) & s.unabsorbed

        s.ch[to_force_absorb] = torch.round(
            (torch.sign(s.dv[to_force_absorb]) / 2 + .5)).long()
        s.rt[to_force_absorb] = t
        s.unabsorbed[to_force_absorb] = False

        return s.ch, s.rt

    def step_sim_state(self, ev_t, t, state):
        """
        @type ev_t: torch.Tensor
        @param ev_t: [trial, dim]
        @type t: double
        @type state: State2DPar
        @return: state
        @rtype: State2DPar
        """

        s = state
        bound = self.get_bound(t)

        ddv, stimulus_ended, log_p_ddv = self.get_ddv(ev_t, t)

        s.dv = s.dv + ddv * s.unabsorbed.double()
        # s.dv[s.unabsorbed] = (
        #     s.dv[s.unabsorbed] + ddv[s.unabsorbed]
        # )

        # log_p_ddv.sum().backward()
        # state.dv.sum().backward()
        # print('--')

        s.log_p_unabs = (
            s.log_p_unabs + log_p_ddv * s.unabsorbed.double()
        )
        # Avoid in-place op to prevent “leaf variable has been moved into
        # the graph interior” error:
        # see: https://discuss.pytorch.org/t/leaf-variable-was-used-in-an-inplace-operation/308/2?u=yulkang
        # s.log_p_unabs[s.unabsorbed] = (
        #     s.log_p_unabs[s.unabsorbed]
        #     + log_p_ddv[s.unabsorbed]
        # ).clone()

        # # Set log_p_unabs to -inf from Td + 1 (rather than from Td)
        s.log_p_unabs[~s.unabsorbed] = -np.inf

        # just for the record
        s.log_p_ddv[~s.unabsorbed] = 0.
        s.log_p_ddv[s.unabsorbed] = log_p_ddv[s.unabsorbed]

        for dim in range(self.n_dim):
            for ch1 in [0, 1]:
                sign_ch = np.sign(ch1 - .5)
                bound1 = bound[None, dim, ch1] * sign_ch
                new_absorbed = (
                    s.dv[:, dim] * sign_ch >= bound1
                ) & s.unabsorbed[:, dim]
                s.dv[new_absorbed, dim] = bound1 * sign_ch
                s.ch[new_absorbed, dim] = ch1
                s.rt[new_absorbed, dim] = t
                s.unabsorbed[new_absorbed, dim] = False

        s.ch, s.rt = self.force_absorb(ev_t, t, s)
        return s

    def simulate(
            self,
            ev,
            store_internals=True,
            return_numpy=True,
    ):
        """
        @type ev: torch.FloatTensor
        @param ev: evidence[trial, dim, time]
        @type n_particle_per_trial: int
        @type store_internals: bool
        @param store_internals: if True, return state at every time step
        @type return_numpy: bool
        @rtype: Tuple[torch.LongTensor, torch.FloatTensor, State2D]
        @return ch, rt, states: ch[trial, dim], rt[trial, dim] (in second)
        """
        if not torch.is_tensor(ev):
            ev = torch.tensor(ev)

        state = self.init_sim_state(ev)
        states = []

        nt = ev.shape[2]
        for it in range(nt):
            t = it * self.dt
            state = self.step_sim_state(ev[:,:,it], t, state)
            if store_internals:
                states.append(state.get_dict_state_at_time(return_numpy))

        ch = state.ch
        rt = state.rt
        if store_internals:
            state.summarize_across_time(states, return_numpy=return_numpy)

        return ch, rt, state

    def summarize_particles(
            self,
            ch, rt, state,
            group=None,
            plot_debug=False
    ):
        """
        Normalize particle weights at each time step among the
        replicates of each trial.

        @type ch: torch.LongTensor
        @type rt: torch.Tensor
        @type state: State2D
        @type group: Union[torch.LongTensor, np.ndarray]
        @rtype: torch.FloatTensor
        @return: p_ch[trial, dim, ch]
        """
        if group is None:
            group = np.arange(ch.shape[0])
        n_tr = np.amax(group) + 1
        n_sim = ch.shape[0]
        n_repeat = n_sim // n_tr

        # log_p_unabs[particle, trial, dim, time]
        log_p_unabs0 = reshape_to_particle_by_trial(
            state.log_p_unabs, n_repeat=n_repeat
        )

        # TODO: check the time course of log_p_unabs in classes other
        #  than parallel
        is0 = log_p_unabs0 == -np.inf
        log_p_unabs = log_p_unabs0 - torch.max(log_p_unabs0, dim=0,
                                               keepdim=True)[0]

        # # Use empty to avoid "one of the variables needed for gradient
        # # computation has been modified by an inplace operation" error
        # p_unabs = torch.exp(log_p_unabs)
        p_unabs = torch.empty_like(log_p_unabs)
        p_unabs[~is0] = torch.exp(log_p_unabs[~is0])
        p_unabs[is0] = 0.


        # P(v_{i, 0:t} | Td >= t)
        #   = p_dv_upto_t_given_unabs[particle, trial, dim, time]
        #   v: decision variable = accumulated evidence
        #   i: particle number (within each trial)
        #   Td: decision time = absorption time
        p_dv_upto_t_given_unabs = npt.sumto1(p_unabs, dim=0)

        # td[particle, trial, dim]
        td = (
            reshape_to_particle_by_trial(state.rt, n_repeat=n_repeat)
            / self.dt
        ).long()
        particle, trial, dim = torch.meshgrid([
            torch.arange(n_repeat, dtype=torch.long),
            torch.arange(n_tr, dtype=torch.long),
            torch.arange(self.n_dim, dtype=torch.long)
        ])

        # P(Td == t | Td >= t)
        #   = p_Td_eq_t_given_Td_gt_t[particle, trial, dim, time]
        p_Td_eq_t_given_Td_gt_t = torch.zeros_like(p_dv_upto_t_given_unabs)


        p_dv_traj_at_Td = p_dv_upto_t_given_unabs[
            particle, trial, dim, td
        ]

        # P(Td = t | Td >= t)
        #   = p_Td_eq_t_given_Td_gt_t[particle, trial, dim, time]
        p_Td_eq_t_given_Td_gt_t[
            particle, trial, dim, td
        ] = p_dv_traj_at_Td

        # P(Td != t | Td >= t)
        #   = p_Td_neq_t[trial, dim, time]
        p_Td_neq_t = 1. - p_Td_eq_t_given_Td_gt_t.sum(0)

        # P(Td > t)
        #   = p_Td_ge_t[trial, dim, time]
        def roll(v, shift=1):
            return torch.cat([
                torch.ones(v.shape[:-1] + torch.Size([1])),
                npt.p2en(npt.p2st(v, 1)[:-1])
            ], -1)
        p_Td_ge_t = roll(torch.cumprod(p_Td_neq_t, -1))

        # P(v_{i, 0:t}, Td >= t)
        #   = p_dv_upto_t_and_unabs[particle, trial, dim, time]
        p_dv_upto_t_and_unabs = (
            p_dv_upto_t_given_unabs * p_Td_ge_t[None, :]
        )

        # P(v_{i, Td}, Td)
        #   = p_dv_at_Td[particle, trial, dim]
        p_dv_at_Td = p_dv_upto_t_and_unabs[
            particle, trial, dim, td
        ]

        # P(ch_dim)
        #   = p_ch[trial, dim, ch]
        p_ch = torch.empty(n_tr, self.n_dim, self.n_ch)

        # ch_by_particle[particle, trial, dim]
        ch_by_particle = reshape_to_particle_by_trial(
            ch, n_repeat=n_repeat
        )
        for ch1 in np.arange(self.n_ch):
            p_ch[:, :, ch1] = (
                p_dv_at_Td * (ch_by_particle == ch1).double()
            ).sum(0)

        # NOTE: plot dv trajectories of particles within the same
        #  trial, along with Td
        trial1 = 7
        dim1 = 0
        if plot_debug:
            nt = p_dv_upto_t_given_unabs.shape[-1]
            t = np.arange(nt) * self.dt

            plt.clf()
            gs = plt.GridSpec(nrows=5, ncols=1)

            ax0 = plt.subplot(gs[0, 0])
            dv = reshape_to_particle_by_trial(state.dv, n_repeat=n_repeat)
            plt.plot(t, npy(dv[:, trial1, dim1, :]).T, '.-')
            plt2.lim_symmetric('y')
            plt2.box_off()
            plt2.hide_ticklabels('x', ax=ax0)
            # plt.setp(ax0.get_xticklabels(), visible=False)
            # plt.gca().set_xticklabels([])
            plt2.detach_axis('x')
            plt.ylabel('$v$')

            plt.subplot(gs[1, 0], sharex=ax0)
            p = npy(p_dv_upto_t_given_unabs[:, trial1, dim1, :]).T
            p[p == 0] = np.nan # to make it clear when arriving at zero
            plt.plot(t, p, '.-')
            td1 = td[:, trial1, dim1]
            repeats = torch.arange(n_repeat, dtype=torch.long)
            p = p_dv_upto_t_given_unabs[repeats, trial1, dim1, td1].T
            t1 = td1.double() * self.dt
            plt2.hide_ticklabels('x')
            plt.plot(*npys(t1, p), 'o')
            plt.ylabel(r'$\mathrm{P}(v_{0:t} \mid T_\mathrm{d} \geq t)$')
            plt2.box_off()
            plt2.detach_axis('x')
            plt.ylim([-0.05, 1.05])
            plt2.detach_axis('y', amin=0, amax=1)

            plt.subplot(gs[2, 0], sharex=ax0)
            p = npy(p_Td_ge_t[trial1, dim1, :]).T
            plt.plot(t, p, '.-')
            plt.plot(t1, p[td1], 'o')
            plt.ylabel(r'$\mathrm{P}(T_\mathrm{d} \geq t)$')
            plt2.box_off()
            plt2.hide_ticklabels('x')
            plt2.detach_axis('x')
            plt.ylim([-0.05, 1.05])
            plt2.detach_axis('y', amin=0, amax=1)

            plt.subplot(gs[3, 0], sharex=ax0)
            p = npy(p_dv_upto_t_and_unabs[:, trial1, dim1, :]).T
            p[p == 0] = np.nan # to make it clear when arriving at zero
            plt.plot(t, p, '.-')
            td1 = td[:, trial1, dim1]
            repeats = torch.arange(n_repeat, dtype=torch.long)
            p = p_dv_upto_t_and_unabs[repeats, trial1, dim1, td1].T
            t1 = td1.double() * self.dt
            plt.plot(*npys(t1, p), 'o')
            plt.ylabel(r'$\mathrm{P}(v_{0:t} \mid T_\mathrm{d} \geq t)$')
            plt2.box_off()
            plt2.detach_axis('x')
            plt.ylim([-0.05, 1.05])
            plt2.detach_axis('y', amin=0, amax=1)
            plt.xlabel('time (s)')

            plt.subplot(gs[4, 0])
            particle1 = 0
            # ecdf is not uniform - but perhaps it's fine when coherence
            # is not zero
            plt2.ecdf(npy(p_dv_at_Td[particle1, :100, dim]).flatten())
            plt.xlabel(r'$\mathrm{P}(v_{T\mathrm{d}})$')
            plt2.box_off()
            plt2.detach_axis()

            plt.show()

            print(t1)
            print(p)
            print('--')

            # NOTE: Check timepoint-wise normalization across particles
            #   within each trial
            # log_p_unabs_example = npy(log_p_unabs0[:, 0, 0, :]).T
            # print(log_p_unabs_example[-1, :])
            # plt.plot(log_p_unabs_example)
            # plt.ylabel('log p unabs')
            # plt.xlabel('time (frame)')
            # plt.show()

            # max_err = torch.max(torch.abs(p_unabs.sum(0) - 1))
            # print(max_err)
            #
            # log_p_unabs_example = npy(log_p_unabs0[0, :100, 0, :])
            # gs = plt.GridSpec(nrows=1, ncols=4)
            # plt.subplot(gs[0, 0])
            # plt.title('log_p')
            # plt.imshow(log_p_unabs_example)
            #
            # plt.subplot(gs[0, 1])
            # plt.imshow(np.isnan(log_p_unabs_example))
            # plt.title('isnan')
            #
            # plt.subplot(gs[0, 2])
            # plt.imshow(np.isinf(log_p_unabs_example))
            # plt.title('isinf')
            #
            # plt.subplot(gs[0, 3])
            # plt.imshow(log_p_unabs_example < 0)
            # plt.title('<0')
            #
            # plt.show()
            #
            # assert torch.all(max_err < 1e-6)

        return p_ch

    def forward(
            self,
            ev,
            n_particle_per_trial=1,
            group=None,
            return_by_particle=False
    ):
        """
        @param ev: [trial, dim, time]
        @type ev: torch.Tensor
        @type n_particle: int
        @return: p_ch[trial, dim, ch]
        @rtype: torch.Tensor
        """
        n_tr0 = ev.shape[0]
        ev = ev.repeat([n_particle_per_trial]
                       + [1] * (ev.ndimension() - 1))

        ch, rt, state = self.simulate(
            ev,
            store_internals=True,
            return_numpy=False,
        )

        p_ch = self.summarize_particles(
            ch, rt, state,
            group=repeat_trials(np.arange(n_tr0),
                                n_repeat=n_particle_per_trial)
        )
        return p_ch


class State2DSer(State2DPar):
    pass


class Dtb2DSerSim(Dtb2DParSim):
    """
    Start with one random dimension first, ignoring evidence
    """
    def __init__(
            self,
            *args,
            switch_rule='alternate',
            alternate_dim_every=np.inf,
            **kwargs
    ):
        """

        @param args:
        @param switch_rule: 'alternate', 'random', 'prioritize'
        @type alternate_dim_every: double
        @param alternate_dim_every: in seconds
        @param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.model_kind = 'serial'
        self.state_class = State2DSer
        self.switch_rule = switch_rule

        self.alternate_dim_every = torch.tensor(alternate_dim_every)
        self.p_dim_1st = ykt.ProbabilityParameter([.5, .5])
        # self.register_probability_parameter('p_dim_1st', [0.5, 0.5])

    def init_sim_state(self, ev):
        """

        @param ev:
        @rtype: State2DSer
        """
        s = super().init_sim_state(ev)  # type: State2DSer

        dim_open = npt.categrnd(
            self.p_dim_1st[None, :].expand(s.n_tr, -1)
        ).flatten().long()

        s.dv_open = torch.zeros(s.n_tr, consts.N_DIM, dtype=torch.bool)
        s.dv_open[torch.arange(s.n_tr), dim_open] = True

        return s

    def switch_dv_open(self, ev_t, t, state, inplace=True):
        if inplace:
            dv_open = state.dv_open
        else:
            dv_open = state.dv_open.clone()
        if t > 0 and (
                torch.floor(t / self.alternate_dim_every)
                > torch.floor((t - self.dt) / self.alternate_dim_every)
        ):  # if a period has passed
            all_unabs = state.unabsorbed.all(1)
            if self.switch_rule == 'alternate':
                # switch all unabsorbed
                dv_open[all_unabs] = ~dv_open[all_unabs]

            elif self.switch_rule == 'random':
                # determine dim according to p_dim_1st
                n_all_unabs = all_unabs.sum()
                dv_open_prev = dv_open.clone()
                dv_open[all_unabs] = False
                dim_open = npt.categrnd(
                    self.p_dim_1st[None, :].expand(n_all_unabs, -1)
                ).flatten().long()
                torch.nonzero(all_unabs)
                dv_open[torch.nonzero(all_unabs)[:, 0], dim_open] = True
                # print(dv_open_prev[all_unabs].double().mean(0))
                # print(dv_open[all_unabs].double().mean(0))
                # print('--')

            elif self.switch_rule == 'prioritize':
                # determine dim according to ev_t
                raise NotImplementedError()

            else:
                raise ValueError()
        return dv_open

    def step_sim_state(self, ev_t, t, state):
        """
        @param ev_t:
        @param t:
        @type state: State2DSerBuffer
        @return:
        """
        s = state
        bound = self.get_bound(t)
        s.dv_open = self.switch_dv_open(ev_t, t, state)

        ddv, stimulus_ended, log_p_ddv = self.get_ddv(ev_t, t)
        s.log_p_ddv.zero_()

        dv_open_new = s.dv_open.clone()
        for dim in range(self.n_dim):
            incl = s.dv_open[:, dim] & s.unabsorbed[:, dim]
            s.dv[incl, dim] = s.dv[incl, dim] + ddv[incl, dim]

            for ch1 in [0, 1]:
                sign_ch = np.sign(ch1 - .5)
                bound1 = bound[None, dim, ch1] * sign_ch
                new_absorbed = (
                    s.dv[:, dim] * sign_ch >= bound1
                ) & s.unabsorbed[:, dim]
                s.ch[new_absorbed, dim] = ch1
                s.rt[new_absorbed, dim] = t
                s.dv[new_absorbed, dim] = bound1 * sign_ch

                s.unabsorbed[new_absorbed, dim] = False
                odim = 1 - dim
                dv_open_new[new_absorbed, dim] = False
                dv_open_new[new_absorbed & s.unabsorbed[:, odim], odim] =\
                    True

        s.dv_open = dv_open_new
        s.ch, s.rt = self.force_absorb(ev_t, t, s)

        return s


class State2DSerBuffer(State2DSer):
    def __init__(self, ev):
        super().__init__(ev)
        self.key_state_at_time.update({
            'buffered_ev', 'buffer_dur_left', 'buffer_open',
            'time_aft_stim_offset'
        })
        self.buffered_ev = torch.zeros_like(self.dv)
        self.buffer_dur_left = torch.zeros(self.dv.shape, dtype=torch.long)
        self.buffer_open = torch.ones_like(self.dv_open)
        self.log_p_dbufv = torch.zeros_like(self.log_p_ddv)
        self.time_aft_stim_offset = torch.zeros_like(self.dv)

class Dtb2DSerBufferSim(Dtb2DSerSim):
    """
    Start with one random dimension first, ignoring evidence
    """
    def __init__(
            self,
            *args,
            buffer_dur=0.12,
            alternate_dim_every=None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.model_kind = 'serial_w_buffer'
        self.state_class = State2DSerBuffer

        self.buffer_dur = ykt.BoundedParameter(
            [buffer_dur], 0.05, 0.5
        )
        # self.register_bounded_parameter(
        #     'buffer_dur', buffer_dur,
        #     0.05, 0.5
        # )
        # ddv_per_buffer_ev: controls decay
        self.ddv_per_buffer_ev = ykt.BoundedParameter(
            0.99 + torch.zeros(consts.N_DIM), # torch.ones(consts.n_dim) /
            # self.buffer_dur,
            0.01, 0.99
        )
        # self.register_bounded_parameter(
        #     'ddv_per_buffer_ev', 0.5 + torch.zeros(consts.n_dim),
        #     0.01, 0.99
        # )
        if alternate_dim_every is None:
            self.alternate_dim_every = self.buffer_dur[0]
        else:
            self.alternate_dim_every = torch.tensor(alternate_dim_every)
            # raise NotImplementedError()


    def init_sim_state(self, ev):
        """
        @rtype: State2DSerBuffer
        """
        s = super().init_sim_state(ev)  # type: State2DSerBuffer
        s.buffered_ev = torch.zeros_like(s.dv)
        s.buffer_dur_left = \
            self.buffer_dur.v + torch.zeros_like(s.dv)
        s.buffer_open = torch.ones_like(s.dv_open)
        s.time_aft_stim_offset = -torch.ones_like(s.dv)
        return s

    def force_absorb(self, ev_t, t, s):
        """

        @param ev_t:
        @param t:
        @type s: State2DSerBuffer
        @return:
        """
        # TODO: limit range of dv within bound
        #   for dim in range(self.n_dim):
        #     s.dv[s.unabsorbed[:,dim], dim]._clamp(
        #         min=self.get_bound(t)[]
        #     )

        # Force absorb after buffer_dur passed from stimulus offset
        if t == (s.nt - 1) * self.dt:
            to_force_absorb = s.unabsorbed
        else:
            to_force_absorb = s.unabsorbed & (
                s.time_aft_stim_offset >= 0.
                # s.time_aft_stim_offset >= self.buffer_dur * 2
            )

        # First transfer remaining dv
        s.dv[to_force_absorb] = (
            s.dv[to_force_absorb]
            + s.buffered_ev[to_force_absorb]
        )
        s.buffered_ev[to_force_absorb] = 0.

        # Then absorb
        s.ch[to_force_absorb] = torch.round(
            (torch.sign(s.dv[to_force_absorb]) / 2 + .5)).long()
        s.rt[to_force_absorb] = t
        s.unabsorbed[to_force_absorb] = False

        return s.ch, s.rt

    def step_sim_state(self, ev_t, t, state):
        """
        @param ev_t:
        @param t:
        @type state: State2DSerBuffer
        @return:
        """
        s = state
        bound = self.get_bound(t)
        s.dv_open = self.switch_dv_open(ev_t, t, state)

        d_bufv, stimulus_ended, log_p_dbufv = self.get_ddv(ev_t, t)
        s.time_aft_stim_offset = (
            s.time_aft_stim_offset
            + self.dt
        )
        s.time_aft_stim_offset[
            torch.isnan(s.time_aft_stim_offset) & stimulus_ended
            ] = 0.
        s.log_p_dbufv.zero_()

        for dim in range(self.n_dim):
            incl = s.buffer_open[:, dim]
            s.buffered_ev[incl, dim] = s.buffered_ev[incl, dim] + (
                d_bufv[incl, dim]
                * (s.buffer_dur_left[incl, dim] / self.dt).clamp_max(1.)
            )
            s.buffer_dur_left[incl, dim] = (
                    s.buffer_dur_left[incl, dim] - self.dt
            ).clamp_min(0.)
            s.log_p_dbufv[incl, dim] = log_p_dbufv[incl, dim]

            # close buffer if buffer duration ran out
            s.buffer_open[:, dim] = s.buffer_dur_left[:, dim] > 0.

        dv_open_new = s.dv_open.clone()
        for dim in range(self.n_dim):
            incl = s.dv_open[:, dim] & s.unabsorbed[:, dim]
            ddv = s.buffered_ev[incl, dim] * self.ddv_per_buffer_ev[dim]
            s.dv[incl, dim] = s.dv[incl, dim] + ddv
            s.buffered_ev[incl, dim] = s.buffered_ev[incl, dim] - ddv
            s.buffer_dur_left[incl, dim] = (
                    s.buffer_dur_left[incl, dim]
                + self.dt
            )

            for ch1 in [0, 1]:
                sign_ch = np.sign(ch1 - .5)
                bound1 = bound[None, dim, ch1] * sign_ch
                new_absorbed = (
                    s.dv[:, dim] * sign_ch >= bound1
                ) & s.unabsorbed[:, dim]
                s.ch[new_absorbed, dim] = ch1
                s.rt[new_absorbed, dim] = t
                s.dv[new_absorbed, dim] = bound1 * sign_ch

                s.unabsorbed[new_absorbed, dim] = False
                odim = 1 - dim
                dv_open_new[new_absorbed, dim] = False
                dv_open_new[new_absorbed & s.unabsorbed[:, odim], odim] =\
                    True

            # if torch.any(torch.abs(s.dv) > torch.abs(bound[None, :, 0])):
            #     print('--')

        s.dv_open = dv_open_new
        s.ch, s.rt = self.force_absorb(ev_t, t, s)

        # if torch.any(torch.abs(s.dv) > torch.abs(bound[None, :, 0])):
        #     print('--')

        return s


#%%
model_classes = [
    Dtb2DParSim, Dtb2DSerSim, Dtb2DSerBufferSim
]
model_kinds = [m().model_kind for m in model_classes]


def get_model(kind, *args, **kwargs):
    """
    @type kind: str
    @rtype: Dtb2DParSim
    """
    return model_classes[model_kinds.index(kind)](*args, **kwargs)