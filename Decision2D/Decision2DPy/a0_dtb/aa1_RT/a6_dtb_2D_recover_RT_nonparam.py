#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

from collections import OrderedDict as odict
from copy import deepcopy
from typing import Union, Type, List, Dict, Iterable, Tuple

import numpy as np
import numpy_groupies as npg
import torch
from matplotlib import pyplot as plt

from a0_dtb import a3_dtb_2D_sim as sim2d
from a0_dtb.aa1_RT import a5_dtb_2D_fit_RT_nonparam as np2d
from data_2d import consts
from lib.pylabyk import localfile, np2, plt2

locfile = localfile.LocalFile(
    pth_root='../../Data_2D/Data_2D_Py/a0_dtb/RTRecover',
    cache_dir=''
)

dtb2ds = [sim2d.RTNonparam2DSer, sim2d.RTNonparam2DPar]

max_epoch = 300
# max_epoch = 1  # CHECKED

to_plot_progress = False  # CHECKED

# subj_parad_bis0 = [consts.SUBJ_PARAD_BI[k]
#                    for k in [
#                        # 1,
#                        7,
#                        # 13
#                    ]]

subj_parad_bis0 = consts.SUBJ_PARAD_BI
# subj_parad_bis0 = consts.SUBJ_PARAD_BI[1:3]  # CHECKED
# subj_parad_bis0 = consts.SUBJ_PARAD_BI[-1::-1]  # CHECKED

td_short = ['serial', 'parallel']
td_fits_short = ['ser_np', 'par_np']
td_sims_short = ['ser_np', 'par_np']

# preset1 = 'thtr=10+trst=200+nfl=5+sm1=0+lpsub=1e-3+dsk=sd+dsub=4'
# preset1 = 'thtr=10+trst=200+nfl=5+sm1=1+lpsub=1e-6+dsk=sd'
# preset1 = 'thtr=10+trst=200+nfl=5+sm1=0+lpsub=1e-3+dsk=sd'
# preset1 = 'thtr=10+trst=200+nfl=5+sm1=1+lpsub=1e-3+dsk=sd'
# preset1 = 'co=1+nfl=5'
# preset1 = 'co=1+nfl=5+lpsub=1e-6+sm1=1+dspub=3'
# preset1 = 'co=1+nfl=5+lpsub=1e-6+sm1=0+dspub=3'
# preset1 = 'co=1+nfl=5+lpsub=1e-6+sm1=0+dspub=0.99'
# preset1 = 'co=1+nfl=5+lpsub=1e-6+sm1=0+dspub=3'
# preset1 = 'co=1+nfl=5+lpsub=1e-3+sm1=0+dspub=0.99'
# preset1 = 'co=1+nfl=5+lpsub=1e-3+sm1=1+dspub=0.99'
preset1 = 'co=1+nfl=5+lpsub=1e-3+sm1=0+dspub=0.95'
# preset1 = 'co=1+nfl=5+lpsub=1e-3+sm1=1+dspub=0.95'
# preset1 = 'co=1+nfl=5+lpsub=1e-6+sm1=1+dspub=3'

preset_recovery = odict([(
    'co=1+nfl=5+lpsub=1e-3+sm1=1+dspub=0.95', {
        'preset_label': 'correct only\n5-fold crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': True,
        'n_fold_test': 5,
        'mode_train': 'all',
        'disper_ub': 0.95,
        'sumto1_wi_cond': True,
        'lapse_max': 1e-3,
    }), (
    'co=1+nfl=5+lpsub=1e-3+sm1=0+dspub=0.95', {
        'preset_label': 'correct only\n5-fold crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': True,
        'n_fold_test': 5,
        'mode_train': 'all',
        'disper_ub': 0.95,
        'sumto1_wi_cond': False,
        'lapse_max': 1e-3,
    }), (
    'co=1+nfl=5+lpsub=1e-3+sm1=1+dspub=0.99', {
        'preset_label': 'correct only\n5-fold crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': True,
        'n_fold_test': 5,
        'mode_train': 'all',
        'disper_ub': 0.99,
        'sumto1_wi_cond': True,
        'lapse_max': 1e-3,
    }), (
    'co=1+nfl=5+lpsub=1e-3+sm1=0+dspub=0.99', {
        'preset_label': 'correct only\n5-fold crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': True,
        'n_fold_test': 5,
        'mode_train': 'all',
        'disper_ub': 0.99,
        'sumto1_wi_cond': False,
        'lapse_max': 1e-3,
    }), (
    'co=1+nfl=5+lpsub=1e-6+sm1=0+dspub=3', {
        'preset_label': 'correct only\n5-fold crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': True,
        'n_fold_test': 5,
        'mode_train': 'all',
        'disper_ub': 3.,
        'sumto1_wi_cond': False,
        'lapse_max': 1e-6,
    }), (
    'co=1+nfl=5+lpsub=1e-6+sm1=1+dspub=3', {
        'preset_label': 'correct only\n5-fold crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': True,
        'n_fold_test': 5,
        'mode_train': 'all',
        'disper_ub': 3.,
        'sumto1_wi_cond': True,
        'lapse_max': 1e-6,
    }), (
    'co=1+nfl=5+lpsub=1e-6+sm1=0+dspub=0.99', {
        'preset_label': 'correct only\n5-fold crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': True,
        'n_fold_test': 5,
        'mode_train': 'all',
        'disper_ub': 0.99,
        'sumto1_wi_cond': False,
        'lapse_max': 1e-6,
    }), (
    'co=1+nfl=5+lpsub=1e-6+sm1=0+dspub=3', {
        'preset_label': 'correct only\n5-fold crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': True,
        'n_fold_test': 5,
        'mode_train': 'all',
        'disper_ub': 3.,
        'sumto1_wi_cond': False,
        'lapse_max': 1e-6,
    }), (
    'co=1+nfl=5+lpsub=1e-3+sm1=0+dspub=3', {
        'preset_label': 'correct only\n5-fold crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': True,
        'n_fold_test': 5,
        'mode_train': 'all',
        'disper_ub': 3.,
        'sumto1_wi_cond': False,
        'lapse_max': 1e-3,
    }), (
    'co=1+nfl=5+lpsub=1e-6+sm1=1', {
        'preset_label': 'correct only\n5-fold crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': True,
        'n_fold_test': 5,
        'mode_train': 'all',
        'sumto1_wi_cond': True,
        'lapse_max': 1e-6,
    }), (
    'co=1+nfl=5+lpsub=1e-3+sm1=0', {
        'preset_label': 'correct only\n5-fold crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': True,
        'n_fold_test': 5,
        'mode_train': 'all',
        'sumto1_wi_cond': False,
        'lapse_max': 1e-3,
    }), (
    'co=1+e0=1+nfl=5', {
        'preset_label': 'correct only,\nexcl 0-coh\n5-fold crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': True,
        'correct_only': True,
        'n_fold_test': 5,
        'mode_train': 'all',
    }), (
    'thtr=10+trst=200+nfl=5+sm1=0+lpsub=1e-3+dsk=sd+dsub=4', {
        'preset_label': '# 10 tr, skip first 200,\nmed lapse, '
                        'high disper\n5-fold crossval',
        'trial_st': 200,
        'thres_n_tr': 10,
        'exclude_0coh': False,
        'correct_only': False,
        'n_fold_test': 5,
        'mode_train': 'all',
        'sumto1_wi_cond': True,
        'disper_kind': 'sd',
        'disper_ub': 4.,
        'lapse_max': 1e-3,
    }), (
    'thtr=10+trst=200+nfl=5+sm1=1+lpsub=1e-6+dsk=sd', {
        'preset_label': '# 10 tr, skip first 200,\nmed lapse, '
                        'high disper\n5-fold crossval',
        'trial_st': 200,
        'thres_n_tr': 10,
        'exclude_0coh': False,
        'correct_only': False,
        'n_fold_test': 5,
        'mode_train': 'all',
        'sumto1_wi_cond': True,
        'disper_kind': 'sd',
        'disper_ub': 2.,
        'lapse_max': 1e-6,
    }), (
    'thtr=10+trst=200+nfl=5+sm1=0+lpsub=1e-3+dsk=sd', {
        'preset_label': '# 10 tr, skip first 200,\nmed lapse, '
                        'high disper\n5-fold crossval',
        'trial_st': 200,
        'thres_n_tr': 10,
        'exclude_0coh': False,
        'correct_only': False,
        'n_fold_test': 5,
        'mode_train': 'all',
        'sumto1_wi_cond': False,
        'disper_kind': 'sd',
        'disper_ub': 2.,
        'lapse_max': 1e-3,
    }), (
    'thtr=10+trst=200+nfl=5+sm1=1+lpsub=1e-3+dsk=sd', {
        'preset_label': '# 10 tr, skip first 200,\nmed lapse, '
                        'high disper\n5-fold crossval',
        'trial_st': 200,
        'thres_n_tr': 10,
        'exclude_0coh': False,
        'correct_only': False,
        'n_fold_test': 5,
        'mode_train': 'all',
        'sumto1_wi_cond': True,
        'disper_kind': 'sd',
        'disper_ub': 2.,
        'lapse_max': 1e-3,
    }), (
    'thtr=10+trst=200+nfl=5+sm1=1+lpsub=1e-3', {
        'preset_label': '# 10 tr, skip first 200,\nmed lapse, '
                        'high disper\n5-fold crossval',
        'trial_st': 200,
        'thres_n_tr': 10,
        'exclude_0coh': False,
        'correct_only': False,
        'n_fold_test': 5,
        'mode_train': 'all',
        'sumto1_wi_cond': True,
        'disper_ub': 2.,
        'lapse_max': 1e-3,
    }), (
    'thtr=10+trst=200+nfl=5+sm1=1+lpsub=1e-6', {
        'preset_label': '# 10 tr, skip first 200,\nlow lapse, '
                        'high disper\n5-fold crossval',
        'trial_st': 200,
        'thres_n_tr': 10,
        'exclude_0coh': False,
        'correct_only': False,
        'n_fold_test': 5,
        'mode_train': 'all',
        'sumto1_wi_cond': True,
        'disper_ub': 2.,
        'lapse_max': 1e-6,
    }), (
    'co=1+nfl=5+sm1=0+lpsub=1e-6', {
        'preset_label': '# 10 tr, skip first 200,\nlow lapse, '
                        'high disper\n5-fold crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': True,
        'n_fold_test': 5,
        'mode_train': 'all',
        'sumto1_wi_cond': False,
        'disper_ub': 2.,
        'lapse_max': 1e-6,
    }), (
    'thtr=10+trst=200+nfl=5+sm1=0+lpsub=1e-6', {
        'preset_label': '# 10 tr, skip first 200,\nlow lapse, '
                        'high disper\n5-fold crossval',
        'trial_st': 200,
        'thres_n_tr': 10,
        'exclude_0coh': False,
        'correct_only': False,
        'n_fold_test': 5,
        'mode_train': 'all',
        'sumto1_wi_cond': False,
        'disper_ub': 2.,
        'lapse_max': 1e-6,
    }), (
    'thtr=10+trst=200+nfl=1+sm1=0+lpsub=1e-6+mtrn=easiest', {
        'preset_label': '# 10 tr, skip first 200,\nlow lapse, '
                        'high disper, sum to 1\neasiest',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': False,
        'n_fold_test': 1,
        'mode_train': 'easiest',
        'sumto1_wi_cond': False,
        'disper_ub': 2.,
        'lapse_max': 1e-6,
    }), (
    'thtr=10+trst=200+nfl=1+sm1=0+lpsub=1e-6', {
        'preset_label': '# 10 tr, skip first 200,\nlow lapse, '
                        'high disper, sum to 1\nno crossval',
        'trial_st': 200,
        'thres_n_tr': 10,
        'exclude_0coh': False,
        'correct_only': False,
        'n_fold_test': 1,
        'mode_train': 'all',
        'sumto1_wi_cond': False,
        'disper_ub': 2.,
        'lapse_max': 1e-6,
    }), (
    'thtr=10+trst=200+nfl=1+sm1=1+lpsub=1e-6', {
        'preset_label': '# 10 tr, skip first 200,\nlow lapse, '
                        'high disper, sum to 1\nno crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': False,
        'n_fold_test': 1,
        'mode_train': 'all',
        'sumto1_wi_cond': True,
        'disper_ub': 2.,
        'lapse_max': 1e-6,
    }), (
    'thtr=10+trst=200+nfl=1+sm1=1+lpsub=1e-3', {
        'preset_label': '# 10 tr, skip first 200,\nlow lapse, '
                        'high disper, sum to 1\nno crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        'correct_only': False,
        'n_fold_test': 1,
        'mode_train': 'all',
        'sumto1_wi_cond': True,
        'disper_ub': 2.,
        'lapse_max': 1e-3,
    }), (
    'thtr=10+trst=200+nfl=1+sm1=1+lpsub=1e-6', {
        'preset_label': '# 10 tr, skip first 200,\nlow lapse, '
                        'high disper, sum to 1\nno crossval',
        'trial_st': 200,
        'thres_n_tr': 10,
        'exclude_0coh': False,
        'correct_only': False,
        'n_fold_test': 1,
        'mode_train': 'all',
        'sumto1_wi_cond': True,
        'disper_ub': 2.,
        'lapse_max': 1e-6,
    }), (
    'co=1+nfl=1+lpsub=1e-6+sm1=0', {
        'preset_label': 'correct only\nno crossval',
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': False,
        # 'sumto1_wi_cond': False,
        'correct_only': True,
        'n_fold_test': 1,
        'mode_train': 'all',
        'sumto1_wi_cond': False,
        'lapse_max': 1e-6,
    }), (
    'thtr=10+trst=200+nfl=1', {
        'preset_label': 'thres 10 tr,\nskip first 200 tr\nno crossval',
        'trial_st': 200,
        'thres_n_tr': 10,
        'exclude_0coh': False,
        'correct_only': False,
        'n_fold_test': 1,
        'mode_train': 'all',
    }), (
    'thtr=10+trst=200+nfl=5', {
        'preset_label': 'thres 10 tr,\nskip first 200 tr\n5-fold crossval',
        'trial_st': 200,
        'thres_n_tr': 10,
        'exclude_0coh': False,
        'correct_only': False,
        'n_fold_test': 5,
        'mode_train': 'all',
    })])


def get_subj_parad_bi_str(subj_parad_bis
                          : Iterable[Tuple[str, str, bool]] = None):
    """

    :param subj_parad_bis: [('subj', 'parad', is_bimanual), ...]
    :return:
    """
    ss = []
    for subj, parad, bimanual in subj_parad_bis:
        if parad in ['RT', 'eye']:
            s = 'eye, %s' % subj
        elif parad == 'bimanual' or bimanual:
            s = 'bimanual, %s' % subj
        elif parad == 'unibimanual' and not bimanual:
            s = 'unimanual, %s' % subj
        else:
            s = '%s, %s' % (parad, subj)

        ss.append(s)
    return ss


parad_bis, ix_parad_bi = np.unique(
    np.stack([v[1:] for v in subj_parad_bis0]), axis=0,
    return_inverse=True)
colors_parad = {
    ('RT', 'False'): 'tab:orange',
    ('unibimanual', 'False'): 'tab:blue',
    ('unibimanual', 'True'): 'tab:cyan',
    ('binary', 'False'): 'plum',
}
labels_parad = {
    ('RT', 'False'): 'eye',
    ('unibimanual', 'False'): 'unimanual',
    ('unibimanual', 'True'): 'bimanual',
    ('binary', 'False'): 'binary',
}


def ____Compare_recovery_methods____():
    pass


def main_compare_recovery_methods(
):
    recovery_methods = list(preset_recovery.keys())

    # dlosses_by_method[method][seed, data, td_sim]
    dlosses_by_method = odict()
    for name, kw in preset_recovery.items():
        kw1 = deepcopy(kw)
        kw1.pop('preset_label')
        dlosses, td_fits = main_plot_recovery(
            to_plot=False,
            **kw1
        )[:2]
        dlosses_by_method[name] = dlosses

    # --- Scatterplot ---
    axs = plot_scatter_dloss(dlosses_by_method, td_fits)

    file = locfile.get_file_fig('scatter_by_recovery_method',
                                subdir='main_compare_recovery_methods')
    plt.savefig(file, dpi=300)
    print('Saved to %s' % file)

    # --- Bar plot across methods ---
    # NOTE - could add:
    #  (1) mean dloss +- SEM
    #  (2) P(correct sign(dloss))
    # plot_bar_mean_dloss_across_methods(dlosses_by_method, td_fits)

    # --- Bar plot of recovery & model selection within subj ---
    # for recovery_method, kw in enumerate(dlosses_by_method.items()):
    #     plot_bar_dloss_across_subjs(dlosses_by_method, td_fits)

    print('--')


def plot_scatter_dloss(dlosses_by_method, td_fits):
    n_methods = len(preset_recovery)
    axs = plt2.GridAxes(
        1, n_methods,
        widths=1.25, heights=1.25,
        top=0.75, left=1.1, bottom=1.
    )
    td_fits = list(td_fits)
    hs = []
    # dlosses_all = np.stack(v for v in dlosses_by_method.values())
    # max_dloss = np.amax(dlosses_all)
    # min_dloss = np.amin(dlosses_all)
    # d_dloss = max_dloss - min_dloss
    # lim = [min_dloss - d_dloss * 0.05, max_dloss + d_dloss * 0.05]
    for i, (name, kw) in enumerate(preset_recovery.items()):
        ax = axs[0, i]
        plt.sca(ax)
        # plt.xscale('log')
        # plt.yscale('log')

        dloss = dlosses_by_method[name]
        for j, parad_bi in enumerate(parad_bis):
            incl = ix_parad_bi == j
            ser = -dloss[0, incl, td_fits.index('ser_np')] / np.log(10.)
            par = dloss[0, incl, td_fits.index('par_np')] / np.log(10.)

            ser = np.clip(ser, a_min=-10, a_max=10)
            par = np.clip(par, a_min=-10, a_max=10)

            def add_jitter(v, vmax=10):
                incl_jitter = np.abs(v) >= vmax
                v[incl_jitter] = (
                        v[incl_jitter] + np.sign(v[incl_jitter]) *
                        np.random.rand(np.sum(incl_jitter)))
                return v

            ser = add_jitter(ser)
            par = add_jitter(par)

            h = plt.plot(ser, par, '.', color=colors_parad[tuple(parad_bi)])
            plt.axis('square')
            if i == 0:
                hs.append(h[0])

            plt.xticks([-10, 0, 10], [r'$\leq$-10', '0', r'$\geq$10'])
            plt.yticks([-10, 0, 10], [r'$\leq$-10', '0', r'$\geq$10'])
        plt.xlim([-11, 11])
        plt.ylim([-11, 11])
        # plt.xlim(lim)
        # plt.ylim(lim)
        plt.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        plt.axvline(0, color='gray', linewidth=0.5, linestyle='--')
        plt2.box_off()
        plt.title(kw['preset_label'])

        if i == 0:
            plt.xlabel('correct support\nfor serial\n'
                       r'($\Delta\mathrm{log}_{10}\mathcal{L}$)')
            plt.ylabel('correct support\nfor parallel\n'
                       r'($\Delta\mathrm{log}_{10}\mathcal{L}$)')
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
    # plt2.sameaxes(axs[:])
    plt.figlegend(hs, [labels_parad[tuple(k)] for k in parad_bis],
                  loc='lower right', frameon=False,
                  handletextpad=0.4
                  )
    for i in range(n_methods):
        plt.sca(axs[0, i])
        plt2.patch_chance_level(1, xy='x')
        plt2.patch_chance_level(1, xy='y')

    return axs


def ____Real_data___():
    pass


def main_plot_real_data(
        mode_train='all',
        n_fold_test=5,
        to_plot=True,
        **kwargs,
):
    """

    :param mode_train:
    :param n_fold_test:
    :param to_plot:
    :param kwargs:
    :return: (
        dlosses, td_fits, losses, ds_cache,
        ix_datas, subj_parad_bis,
        dict_fit_sim, dict_subdir
    )
    """
    if n_fold_test is None:
        if mode_train == 'all':
            n_fold_test = 5
        elif mode_train == 'easiest':
            n_fold_test = 1
        else:
            raise ValueError()

    sbj_str = get_subj_parad_bi_str(subj_parad_bis0)
    cache = locfile.get_cache('mdlcmp', {
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': True,
        'correct_only': True,
        'n_fold_test': n_fold_test,
        'mode_train': mode_train,
        'sbj': '%s-%s' % (
            sbj_str[0], sbj_str[-1]
        ),
        **kwargs
    }, subdir='main_plot_real_data')

    if cache.exists():
        dlosses, td_fits, losses, ds_cache, \
        ix_datas, subj_parad_bis, \
        dict_cache, dict_subdir \
            = cache.getdict([
                'dlosses', 'td_fits', 'losses', 'ds_cache',
                'ix_datas', 'subj_parad_bis',
                'dict_cache', 'dict_subdir'
            ])
    else:
        ds_cache = []
        ds = []
        subj_parad_bis = subj_parad_bis0

        for ix_data, (subj, parad, bimanual) in enumerate(subj_parad_bis):
            for i_fold_test in range(n_fold_test):
                for dtb2d in dtb2ds:
                    # --- Load model fit to real data
                    model, data, dict_cache, dict_subdir, d = np2d.main_fit(
                        dtb2d=dtb2d, subj=subj,
                        parad=parad, bimanual=bimanual,
                        mode_train=mode_train,
                        fit_mode='d_only',
                        # fit_mode='auto',  # CHECKED
                        i_fold_test=i_fold_test,
                        n_fold_test=n_fold_test,
                        max_epoch=max_epoch,
                        to_plot_progress=to_plot_progress,
                        **kwargs,
                    )  # type: (Any, sim2d.Data2DRT, Any, ...)

                    # axs = sim2d.plot_rt_distrib(
                    #     npy(d['out_train_valid']),
                    #     data.ev_cond_dim,
                    #     alpha_face=0.,
                    #     colors=['b', 'b']
                    # )[0]
                    # axs = sim2d.plot_rt_distrib(
                    #     npy(d['target_train_valid']),
                    #     data.ev_cond_dim,
                    #     alpha_face=0.,
                    #     colors=['k', 'k'],
                    #     axs=axs,
                    # )[0]
                    # for ext in ['.pdf', '.png']:
                    #     file = np2d.locfile.get_file_fig(
                    #         'rtdstr', dict_cache, ext=ext, subdir=dict_subdir
                    #     )
                    #     plt.savefig(file, dpi=72, figure=axs[0, 0].figure)
                    #     print('Saved to %s' % file)

                    ds.append(d)
                    ds_cache.append({
                        **dict_cache, **{
                            'ix_data': ix_data
                        }
                    })
                    plt.close('all')

        ds = np2.listdict2dictlist(ds)
        ds_cache = np2.listdict2dictlist(ds_cache)

        ix_data = ds_cache['ix_data']
        ix_datas = np.unique(ix_data)
        # subjs, ix_subj = np.unique(ds_cache['sbj'], return_inverse=True)
        mode_trains, ix_mode_train = np.unique(
            ds_cache['mdtrn'], return_inverse=True)
        td_fits, ix_td_fit = np.unique(ds_cache['td'], return_inverse=True)
        losses_all = np.array(ds['loss_NLL_test'])

        # losses[subj, td_fit]
        losses = npg.aggregate([ix_data, ix_td_fit
        ], losses_all, 'sum')

        # dlosses: [ix_data]: negative supports serial
        dlosses = (losses[:, list(td_fits).index('ser_np')]
                   - losses[:, list(td_fits).index('par_np')])

        cache.set({
            'dlosses': dlosses,
            'td_fits': td_fits,
            'losses': losses,
            'ds_cache': ds_cache,
            'ix_datas': ix_datas,
            'subj_parad_bis': subj_parad_bis,
            'dict_cache': dict_cache,
            'dict_subdir': dict_subdir,
        })
        cache.save()
    del cache

    if to_plot:
        subj_parad_bi_str = get_subj_parad_bi_str(subj_parad_bis)
        dict_file = {
            **dict_cache,
            'td_fit': '[%s]' % ','.join(td_fits_short),
            'sbj': '[%s-%s]' % (subj_parad_bi_str[0], subj_parad_bi_str[-1]),
            'mdtrn': mode_train,
        }
        for subdir in [dict_subdir, 'main_plot_real_data']:
            axs = plot_bar_dloss_across_subjs(dlosses, ix_datas, subj_parad_bis)
            plt.title('Actual Data\n(fit to %s)' % mode_train)

            file = locfile.get_file_fig('dloss_fit_real', dict_file,
                                        subdir=subdir)
            plt.savefig(file, dpi=300)
            print('Saved to %s' % file)

    print('--')
    return (
        dlosses, td_fits, losses, ds_cache,
        ix_datas, subj_parad_bis,
        dict_cache, dict_subdir
    )


def plot_bar_dloss_across_subjs(
        dlosses, elosses=None, ix_datas=None,
        subj_parad_bis: Iterable[Tuple[str, str, bool]] = None,
        axs: Union[plt2.GridAxes, plt2.AxesArray] = None,
        vmax=None,
        add_scale=True,
        base=10.,
):
    """

    :param dlosses: [ix_data]
    :param ix_datas:
    :param axs:
    :param subj_parad_bis: [('subj', 'parad', is_bimanual), ...]
    :return: axs
    """

    if subj_parad_bis is None:
        subj_parad_bis = subj_parad_bis0
    if vmax is None:
        vmax = np.amax(np.abs(dlosses))

    # order: eye S1-S3, hand by ID, paired uni-bimanual
    subjs, parads, bis = zip(*subj_parad_bis)
    subjs = np.array(['ID0' + v[-1] if v[:2] == 'ID' and len(v) == 3 else v
             for v in subjs])
    parads = np.array(parads)
    bis = np.array(bis)

    is_eye = parads == 'RT'
    is_bin = parads == 'binary'
    ix = np.arange(len(subjs))

    def filt_sort(filt):
        # return ix[filt]

        ind = [int(subj[1:]) for subj in subjs[filt]]
        return ix[filt][np.argsort(ind)]

        # return ix[filt][np.argsort(subjs[filt])]

    ix = np.concatenate([
        filt_sort(is_eye & ~is_bin),
        np.stack([filt_sort(~is_eye & ~bis & ~is_bin),
                  filt_sort(~is_eye & bis & ~is_bin)
                  ], -1).flatten('C'),
        filt_sort(is_bin)
    ])
    subjs = subjs[ix]
    parads = parads[ix]
    bis = bis[ix]
    is_eye = is_eye[ix]
    dlosses = dlosses[ix]
    subj_parad_bis = subj_parad_bis[ix]

    n_eye = int(np.sum(is_eye))
    n_hand = int(np.sum(~is_eye))

    y = np.empty([n_eye + n_hand])
    y[is_eye] = 1.5 + np.arange(n_eye)
    y[~is_eye] = n_eye - 1 + 1.5 + np.cumsum([1.5, 1.] * (n_hand // 2))
    y_max = np.amax(y) + 1.5

    if axs is None:
        axs = plt2.GridAxes(
            nrows=1, ncols=1,
            heights=y_max * 0.2, # len(dlosses) * 0.3,
            widths=2,
            left=1.5, right=0.25,
            bottom=0.85
        )
    # print(axs.h)
    ax = axs[0, 0]
    plt.sca(ax)

    m = dlosses
    if elosses is None:
        e = np.zeros_like(m)
    else:
        e = elosses

    for y1, m1, e1, parad1, bi1 in zip(y, m, e, parads, bis):
        plt.barh(y1, m1, xerr=e1,
                 color=colors_parad[(parad1, '%s' % bi1)],
                 edgecolor='None')

    if add_scale:
        dy = y[1] - y[0]
        # for x, ha in zip([-vmax, vmax], ['left', 'right']):
        #     plt.plot([x, x], y[-1] + np.array([-0.5, 0.5]) * dy,
        #              linestyle=':', color='gray', linewidth=0.5)
        #     plt.text(x * 0.95, y[-1], r'$%s\mathrm{log}\,10^{%g}$'
        #              % ('-' if x == -vmax else '+', np.log10(vmax)),
        #              va='center', ha=ha)

    # max_loss = np.amax(np.abs(m) + e)
    # if ix_datas is None:
    #     ix_datas = np.arange(len(dlosses))
    # y = ix_datas

    # plt.barh(y, m / np.log(10),
    #          xerr=e,
    #          color='w',
    #          edgecolor='k')


    # plt2.patch_chance_level(1, xy='x')
    # plt.axvline(0, color='gray', linewidth=0.5, linestyle='--', zorder=-1)

    axvline_dcost()

    # plt.xlim(np.array([-max_loss, max_loss]) * 1.05)
    # plt.xlabel('Support\nfor parallel')

    # ix_big = np.nonzero([tuple(v) == ('ID7', 'unibimanual', True) for v
    #                  in subj_parad_bis])[0][0]
    x_lim = [-vmax * 1.2, vmax * 1.2]
    for ix_big in range(len(y)):
        if np.abs(m[ix_big]) > vmax:
            for i_sign, sign in enumerate([1, -1]):
                plt2.patch_wave(y[ix_big], x_lim[i_sign] * 1.01,
                                ax=ax, color='w',
                                wave_margin=0.15,
                                wave_amplitude=sign * 0.025,
                                )

    plt.xlim(x_lim)
    plt.ylim([y_max - 1., 1.])
    # plt.ylim([-0.75, len(y) - 0.25])
    xticks_serial_vs_parallel(vmax, base)
    subj_parad_bi_str = get_subj_parad_bi_str(subj_parad_bis)
    plt.yticks(y, subj_parad_bi_str)
    plt2.detach_axis('y', y[0], y[-1])
    plt2.detach_axis('x', -vmax, vmax)

    # # CHECKED
    # file = '../../Data_2D/Data_2D_Py/a0_dtb/RTNonparamMATLAB/model_comp.png'
    # plt.savefig(file, dpi=300)
    # axs = None
    # print('Saved to %s' % file)

    return axs


def xticks_serial_vs_parallel(vmax, base):
    plt.xticks([-vmax, 0, vmax])
    plt.xlabel('support for parallel model\n'
               + r'($\mathrm{log}_{%g}\mathrm{BF}$)' % base)

    # plt.xticks([-vmax, 0, vmax], [r'$\leftarrow$' + '\nserial', '',
    #                               r'$\rightarrow$' + '\nparallel'])
    # plt.xlabel('support for model\n' + r'(${\Delta}\mathrm{log}\,\mathcal{L}$)',
    #            labelpad=7, )


def axvline_dcost(BF=100., base=10.):
    plt.axvline(0, color='k', linewidth=0.5, linestyle='--', zorder=1)
    for sign in [-1, 1]:
        plt.axvline(sign * np.log(BF) / np.log(base),
                    color='silver',
                    linewidth=0.5,
                    linestyle='--', zorder=1)
    plt2.box_off()


def ____Simulated_data____():
    pass


def main_plot_recovery(
        mode_train='all',
        n_fold_test=None,
        to_plot=True,
        **kwargs,
) -> (np.ndarray, List[str], np.ndarray, Dict[str, list]):
    """

    :param mode_train:
    :param n_fold_test:
    :param to_plot:
    :param kwargs:
    :return: (
        dlosses[seed, data, td_sim],
        td_fits[model]: str,
        losses[seed, subj, td_sim, td_fit],
        ds_cache[field][ix_data],
        td_sims, ix_datas, subj_parad_bi, seed_sim,
        dict_fit_sim, dict_subdir
    )
    """
    if n_fold_test is None:
        if mode_train == 'all':
            n_fold_test = 5
        elif mode_train == 'easiest':
            n_fold_test = 1
        else:
            raise ValueError()

    sbj_str = get_subj_parad_bi_str(subj_parad_bis0)
    cache = locfile.get_cache('recovery', {
        'trial_st': 0,
        'thres_n_tr': 1,
        'exclude_0coh': True,
        'correct_only': True,
        'n_fold_test': n_fold_test,
        'mode_train': mode_train,
        'sbj': '%s-%s' % (
            sbj_str[0], sbj_str[-1]
        ),
        **kwargs
    }, subdir='main_plot_recovery')

    if cache.exists():
        try:
            dlosses, td_fits, losses, ds_cache, \
            td_sims, ix_datas, subj_parad_bis, seed_sim, \
            dict_fit_sim, dict_subdir \
                = cache.getdict([
                    'dlosses', 'td_fits', 'losses', 'ds_cache',
                    'td_sims', 'ix_datas',
                    'subj_parad_bis',
                    'seed_sim',
                    'dict_fit_sim', 'dict_subdir'
                ])
        except KeyError:  # backward compatibility
            print('subj_parad_bis missing: falling back to old cache for %s'
                  % cache.fullpath)
            dlosses, td_fits, losses, ds_cache, \
            td_sims, ix_datas, _, seed_sim, \
            dict_fit_sim, dict_subdir \
                = cache.getdict([
                'dlosses', 'td_fits', 'losses', 'ds_cache',
                'td_sims', 'ix_datas',
                'subj_parad_bi',
                'seed_sim',
                'dict_fit_sim', 'dict_subdir'
            ])
            subj_parad_bis = subj_parad_bis0
    else:
        ds = []
        ds_cache = []
        subj_parad_bis = subj_parad_bis0

        for ix_data, (subj, parad, bimanual) in enumerate(subj_parad_bis):
            for seed_sim in range(1):
                for i_fold_test in range(n_fold_test):
                    for dtb2d_sim in dtb2ds:
                        for dtb2d_fit in dtb2ds:
                            d, dict_fit_sim, dict_subdir = main_fit_sim(
                                subj=subj,
                                parad=parad,
                                bimanual=bimanual,
                                seed_sim=seed_sim,
                                dtb2d_sim=dtb2d_sim,
                                dtb2d_fit=dtb2d_fit,
                                mode_train=mode_train,
                                n_fold_test=n_fold_test,
                                i_fold_test=i_fold_test,
                                **kwargs,
                            )
                            ds.append(d)
                            ds_cache.append({
                                **dict_fit_sim, **{
                                    'ix_data': ix_data
                                }
                            })
                            plt.close('all')

        ds = np2.listdict2dictlist(ds)
        ds_cache = np2.listdict2dictlist(ds_cache)

        ix_data = ds_cache['ix_data']
        ix_datas = np.unique(ix_data)

        # subjs, ix_subj = np.unique(ds_cache['sbj'], return_inverse=True)
        td_fits, ix_td_fit = np.unique(ds_cache['td_fit'], return_inverse=True)
        losses_all = np.array(ds['loss_NLL_test'])

        seed_sim = np.array(ds_cache['seed_sim'])
        td_sims, ix_td_sim = np.unique(ds_cache['td_sim'], return_inverse=True)

        # losses[seed, subj, td_sim, td_fit]
        # NOTE: aggregate takes care of averaging across i_fold_tests
        losses = npg.aggregate([
            seed_sim, ix_data, ix_td_sim, ix_td_fit
        ], losses_all, 'mean')

        # dlosses: [seed, data, td_sim]: negative supports serial
        dlosses = (losses[:, :, :, list(td_fits).index('ser_np')]
                   - losses[:, :, :, list(td_fits).index('par_np')])

        cache.set({
            'dlosses': dlosses,
            'td_fits': td_fits,
            'losses': losses,
            'ds_cache': ds_cache,
            'td_sims': td_sims,
            'ix_datas': ix_datas,
            'subj_parad_bis': subj_parad_bis,
            'seed_sim': seed_sim,
            'dict_fit_sim': dict_fit_sim,
            'dict_subdir': dict_subdir,
        })
        cache.save()
        del cache

    if to_plot:
        # mean_dlosses: [mode_train, subj, td_sim]
        mean_dlosses = np.mean(dlosses, 0)
        se_dlosses = np2.sem(dlosses, 0)

        axs = plt2.GridAxes(
            nrows=1, ncols=len(td_sims),
            heights=dlosses.shape[1] * 0.3,
            widths=2,
            left=1.5,
            bottom=0.75,
            top=0.7,
        )
        for i_sim in range(len(td_sims)-1, -1, -1):
            col = 1 - i_sim
            plot_bar_dloss_across_subjs(
                mean_dlosses[:, i_sim],
                ix_datas,
                subj_parad_bis,
                axs=axs[:, [col]]
            )
            plt.title('Simulated\n%s' % td_sims[i_sim][:3])

            if col != 0:
                plt2.box_off(['left'])
                plt.yticks([])

        axs.suptitle(mode_train)

        d_file = deepcopy(dict_fit_sim)
        for k in ['sbj', 'prd', 'td_sim', 'td_fit', 'seed_sim']:
            d_file.pop(k)
        for subdir in ['main_plot_recovery', dict_subdir]:
            file = locfile.get_file_fig('dloss_fit', {
                **d_file,
                'tdsm': '[%s]' % ','.join(td_sims_short),
                'tdft': '[%s]' % ','.join(td_fits_short),
                'nsbj': '%d' % len(subj_parad_bis),
                'mdtrn': mode_train,
                'sdsm': '[%g-%g]' % (seed_sim[0], seed_sim[-1]),
            }, subdir=subdir)
            plt.savefig(file, dpi=300)
            print('Saved to %s' % file)

    print('--')
    return (
        dlosses, td_fits, losses, ds_cache,
        td_sims, ix_datas, subj_parad_bis, seed_sim,
        dict_fit_sim, dict_subdir
    )


def main_fit_sim(
        subj='S1',
        parad='RT',
        bimanual=False,
        seed_sim=0,
        dtb2d_sim: Type = sim2d.RTNonparam2DSer,
        dtb2d_fit: Type = sim2d.RTNonparam2DSer,
        mode_train='easiest',
        rt_only=None,
        i_fold_test=0,
        n_fold_test=1,
        **kwargs,
):
    """

    :param subj:
    :param parad:
    :param bimanual:
    :param seed_sim:
    :param dtb2d_sim:
    :param dtb2d_fit:
    :param mode_train:
    :param rt_only:
    :return: d, dict_fit_sim, dict_subdir
    """

    # --- Load model fit to real data
    model, data, dict_cache, dict_subdir, d = np2d.main_fit(
        dtb2d=dtb2d_sim, subj=subj,
        parad=parad, bimanual=bimanual,
        # fit_mode: we may not need to run the model at all if the cached
        #   simulation is available.
        fit_mode='d_only',
        i_fold_test=i_fold_test,
        mode_train=mode_train,
        n_fold_test=n_fold_test,
        to_plot_progress=to_plot_progress,
        **kwargs,
    )

    if rt_only is None:
        rt_only = (
            issubclass(dtb2d_fit, sim2d.RTNonparam2D)
            or isinstance(dtb2d_fit, sim2d.RTNonparam2D))

    # --- Simulate new data (from the model 'fit_sim') and save
    dict_subdir.update({
        'rto': rt_only
    })
    dict_sim = {
        **dict_cache,
        'td_sim': dict_cache['td'],
        'seed_sim': seed_sim,
    }
    dict_sim.pop('td')
    dict_fit_sim = {
        **dict_sim,
        'td_fit': dtb2d_fit.kind
    }
    cache_fit_sim = locfile.get_cache(
        'fit_sim', dict_fit_sim, subdir=dict_subdir)
    if cache_fit_sim.exists():
        best_state, d = cache_fit_sim.getdict([
            'best_state', 'd'
        ])
    else:
        # --- Get/fit the model for simulation
        model, data, dict_cache, dict_subdir, d = np2d.main_fit(
            dtb2d=dtb2d_sim, subj=subj,
            parad=parad, bimanual=bimanual,
            mode_train=mode_train,
            # fit_mode: not 'd_only', since we need d['out_all']
            #    since we need to simulate the data
            fit_mode='auto',
            i_fold_test=i_fold_test,
            n_fold_test=n_fold_test,
            to_plot_progress=to_plot_progress,
            **kwargs,
        )

        # --- Simulate new data and save
        data_sim = deepcopy(data)  # type: sim2d.Data2DRT
        cache_data_sim = locfile.get_cache(
            'data_sim', dict_sim, subdir=dict_subdir)
        if cache_data_sim.exists():
            data_sim.update_data(
                ch_tr_dim=cache_data_sim.getdict(['chSim_tr_dim'])[0],
                rt_tr=cache_data_sim.getdict(['rtSim_tr'])[0]
            )
        else:
            # np2.dict_shapes(d)  # CHECKED
            ch_tr_dim_bef = data_sim.ch_tr_dim.copy()
            rt_tr_bef = data_sim.rt_tr.copy()

            data_sim.simulate_data(
                pPred_cond_rt_ch=d['out_all'],
                seed=seed_sim,
                rt_only=rt_only,  # since nonparam model fits RT only
                )

            ch_tr_dim_aft = data_sim.ch_tr_dim.copy()
            rt_tr_aft = data_sim.rt_tr.copy()

            print('Proportion of trials with the same choice:')
            print(np.mean(ch_tr_dim_bef == ch_tr_dim_aft))

            print('Mean absolute RT difference:')
            print(np.mean(np.abs(rt_tr_bef - rt_tr_aft)))

            cache_data_sim.set({
                'chSim_tr_dim': data_sim.ch_tr_dim,
                'rtSim_tr': data_sim.rt_tr
            })
            cache_data_sim.save()
        del cache_data_sim

        # --- Fit simulated data
        model, data, _, _, d = np2d.main_fit(
            dtb2d=dtb2d_fit, data=data_sim,
            dict_cache=dict_fit_sim,
            dict_subdir=dict_subdir,
            to_save_res=True,
            locfile1=locfile,
            mode_train=mode_train,
            i_fold_test=i_fold_test,
            n_fold_test=n_fold_test,
            max_epoch=max_epoch,
            to_plot_progress=to_plot_progress,
            **kwargs,
        )
        cache_fit_sim.set({
            'best_state': d['best_state'],
            'd': {k: v for k, v in d.items()
                  if k.startswith('loss_')}
        })
        cache_fit_sim.save()
        # # CHECKED
        # print(d['loss_all'])
        # print(best_state['dtb.dtb.dtb1ds.0.kb2._param'])
    del cache_fit_sim

    print('--')
    return d, dict_fit_sim, dict_subdir


def ____Main____():
    pass


if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.double)

    # main_compare_recovery_methods()

    kw1 = deepcopy(preset_recovery[preset1])
    kw1.pop('preset_label')
    main_plot_real_data(**kw1)
    # main_plot_recovery(**kw1)