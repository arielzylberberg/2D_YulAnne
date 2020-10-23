#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

from typing import Union, Iterable
import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
from copy import deepcopy

import torch

from lib.pylabyk.numpytorch import npy, npys
from lib.pylabyk import localfile, argsutil, np2, plt2

from data_2d import consts, load_data
from a0_dtb import a1_dtb_1D_sim as sim1d, a3_dtb_2D_sim as sim2d
from a0_dtb.a1_dtb_1D_sim import save_fit_results

# run "tensorboard --logdir runs" in the terminal to see log

locfile = localfile.LocalFile(
    pth_root='../../Data_2D/Data_2D_Py/a0_dtb/RT',
    cache_dir=''
)

# kw_plot_preds[(serial, parallel)]
kw_plot_preds = [
    {'linestyle': '-', 'linewidth': .5},
    {'linestyle': '--', 'linewidth': .5}
]
kw_plot_data = {
    'markersize': 4.5,
    'mew': 0.5
}

# cmaps[dim_rel]
cmaps = consts.CMAP_DIM


def main_fit_subjs(
        subjs: Iterable[str] = ('S1',),
        parad='RT',  # 'unibimanual'  #
        dtb2ds=(sim2d.Dtb2DRTSerial, sim2d.Dtb2DRTParallel),
        mode_train='all',
        to_plot=('combined', 'rt_distrib'),
        to_group_irr=True,
        ignore_cache=False,
        normalize_ev=True,
        to_combine_ch_irr_cond=True,
        kw_fit=(),
        kw_name=(),
):
    """

    :param subjs:
    :param dtb2ds:
    :param to_plot:
    :param to_group_irr:
    :param kw_fit:
    :param kw_name:
    :return: models, datas
    """
    # to_experiment = False

    if to_group_irr:
        if parad in ['RT', 'short']:
            group_dcond_irr = [[0, 1], [2, 3], [4]]
        elif parad in ['VD']:
            group_dcond_irr = [[0, 1], [2]]
        else:
            group_dcond_irr = [[0, 1], [2, 3, 4], [5]]
    else:
        group_dcond_irr = None

    n_subj = len(subjs)
    n_dtb2d = len(dtb2ds)
    models = np.empty([n_subj, n_dtb2d], dtype=np.object)
    datas = np.empty([n_subj, n_dtb2d], dtype=np.object)
    p_preds = np.empty([n_subj, n_dtb2d], dtype=np.object)
    ev_dense = []
    p_pred_dense = []
    p_datas = np.empty([n_subj, n_dtb2d], dtype=np.object)
    conds = np.empty([n_subj, n_dtb2d], dtype=np.object)
    losses_CE_all = np.empty([n_subj, n_dtb2d], dtype=np.object)
    losses_NLL_all = np.empty([n_subj, n_dtb2d], dtype=np.object)
    losses_CE_test = np.empty([n_subj, n_dtb2d], dtype=np.object)
    losses_NLL_test = np.empty([n_subj, n_dtb2d], dtype=np.object)
    dict_caches = np.empty([n_subj, n_dtb2d], dtype=np.object)

    for ii_subj, subj in enumerate(subjs):
        ev_dense_subj = []
        p_pred_dense_subj = []

        for i_dtb, dtb2d in enumerate(dtb2ds):
            kw_name1 = deepcopy(dict(kw_name))
            kw_fit1 = deepcopy(dict(kw_fit))
            kw_fit1 = argsutil.kwdef(kw_fit1, {
                'tnd_loc_lb': 0.05, # 0.1, #
                'tnd_disper0': 0.2,
                'tnd_disper_ub': 0.5, # 0.3, #
                # 'thres_patience': 1e-4,
                # 'learning_rate': .1,
                'bound0_lb': 0.2,
                'bound_t_st_lb': .1,
                'bound_t_st_ub': 1.0,
                'bound_t_half_lb': .1,
                'bound_asymptote_max': 0.99, # 0.5, #
                'n_fold_valid': 1,
                'subsample_factor': 5,
                'to_normalize_within_cond': True, # NOTE: False in MATLAB
                'trial_st': 0,  # 200,
                'to_allow_irr_ixn': False,
            })
            kw_name1= argsutil.kwdef(kw_name1, {
                'tnd0': kw_fit1['tnd_disper0'],
                'tndub': kw_fit1['tnd_disper_ub'],
                # 'thpt': kw_fit1['thres_patience'],
                # 'lr0': kw_fit1['learning_rate'],
                'btstlb': kw_fit1['bound_t_st_lb'],
                'bthflb': kw_fit1['bound_t_half_lb'],
                'nfldv': kw_fit1['n_fold_valid'],
                # 'subsmp': kw_fit1['subsample_factor'],
            })
            # ---- add to name only if different from default
            for (name0, default, name1) in [
                ('tnd_loc_lb', 0.1, 'tnllb'),
                ('bound0_lb', 0.5, 'blb'),
                ('bound_t_st_ub', 0.8, 'bsu'),
                ('bound_asymptote_max', 0.5, 'baub'),
                ('to_normalize_within_cond', False, 'nrmcnd'),
                ('trial_st', 0, 'trst'),
                ('to_allow_irr_ixn', False, 'ixn')
            ]:
                if kw_fit1[name0] != default:
                    kw_name1[name1] = kw_fit1[name0]

            # CHECKING: debating whether to customize lapse_max for each subj
            # if subj == 'S1':
            #     kw_fit1['lapse_max'] = 0.1
            # elif 'lapse_max' not in kw_fit1:
            #     kw_fit1['lapse_max'] = 1e-3

            model, data, dat, best_loss, dict_cache \
                = main_fit(
                    subj=subj,
                    parad=parad,
                    dtb2d=dtb2d,
                    tnds=sim1d.TndInvNorm,
                    lapse=sim1d.LapseUniform,
                    mode_train=mode_train,  # 'easiest'|'all'
                    ignore_cache=False,  # CHECKED
                    continue_fit=False,  # CHECKED
                    to_debug=False,  # CHECKED
                    kw_name=kw_name1,
                    # patience=100,
                    # learning_rate=0.1,
                    max_epoch=300,  # CHECKED
                    to_compute_grad=False,  # grad not needed for ch-rt plot
                    **kw_fit1
                )
            # model = dtb.FitRT2D()  # CHECKED
            model.train()
            model.zero_grad()

            models[ii_subj, i_dtb] = model
            datas[ii_subj, i_dtb] = data
            dict_caches[ii_subj, i_dtb] = dict_cache

            cache_res = locfile.get_cache('p_pred', dict_cache)
            if cache_res.exists() and not ignore_cache:
                p_pred, p_data, ev_cond_dim, loss_CE_test, loss_NLL_test, \
                loss_CE_all, loss_NLL_all, ev_dense1, p_pred_dense1 = \
                    cache_res.getdict([
                        'p_pred', 'p_data', 'ev_cond_dim',
                        'loss_CE_test', 'loss_NLL_test',
                        'loss_CE_all', 'loss_NLL_all',
                        'ev_dense1', 'p_pred_dense1'
                    ])
            else:
                # --- loss from all data
                ev_cond_fr_dim_meanvar, n_cond_rt_ch = \
                    data.get_data_by_cond('all')[:2]

                ev_cond_dim = data.ev_cond_dim
                p_rt_ch_pred = model(ev_cond_fr_dim_meanvar)
                p_pred = npy(p_rt_ch_pred)
                p_data = n_cond_rt_ch

                loss_CE_all = sim2d.fun_loss(p_rt_ch_pred, p_data,
                                      to_average=True, base_n_bin=True)
                loss_NLL_all = sim2d.fun_loss(p_rt_ch_pred, p_data,
                                      to_average=False, base_n_bin=False)

                # --- loss from test set
                ev_cond_fr_dim_meanvar, p_data = \
                    data.get_data_by_cond('test')[:2]

                p_rt_ch_pred = model(ev_cond_fr_dim_meanvar)
                loss_CE_test = sim2d.fun_loss(p_rt_ch_pred, p_data,
                                      to_average=True, base_n_bin=True)
                loss_NLL_test = npy(sim2d.fun_loss(p_rt_ch_pred, p_data,
                                      to_average=False, base_n_bin=False))

                # --- dense prediction for plotting
                p_pred_dense1 = []
                ev_dense1 = []

                for dim in range(consts.N_DIM):
                    ev_dense11 = sim2d.upsample_ev(ev_cond_fr_dim_meanvar,
                                                   dim_rel=dim)
                    p_pred_dense11 = model(ev_dense11)
                    ev_dense1.append(npy(ev_dense11))
                    p_pred_dense1.append(npy(p_pred_dense11))

                ev_dense1 = np.stack(ev_dense1)
                p_pred_dense1 = np.stack(p_pred_dense1)

                cache_res.set({
                    'p_pred': p_pred,
                    'p_data': p_data,
                    'ev_cond_dim': ev_cond_dim,
                    'loss_CE_all': loss_CE_all,
                    'loss_NLL_all': loss_NLL_all,
                    'loss_CE_test': loss_CE_test,
                    'loss_NLL_test': loss_NLL_test,
                    'ev_dense1': ev_dense1,
                    'p_pred_dense1': p_pred_dense1
                })
                cache_res.save()
            del cache_res
            # del cache to prevent the cache's contents from being
            # manipulated and saved accidentally

            ev_dense_subj.append(ev_dense1)
            p_pred_dense_subj.append(p_pred_dense1)

            # --- store results
            p_preds[ii_subj, i_dtb] = p_pred
            p_datas[ii_subj, i_dtb] = p_data
            conds[ii_subj, i_dtb] = ev_cond_dim

            losses_CE_all[ii_subj, i_dtb] = npy(loss_CE_all)
            losses_NLL_all[ii_subj, i_dtb] = npy(loss_NLL_all)

            losses_CE_test[ii_subj, i_dtb] = loss_CE_test
            losses_NLL_test[ii_subj, i_dtb] = loss_NLL_test

            ev_dense.append(ev_dense1)
            p_pred_dense.append(p_pred_dense1)

            if 'combined' in to_plot:
                # loss_NLL_test.backward()
                axs = None
                to_plot_internals = True
                for to_plot_params in [False]: # , True]:
                    axs = plot_fit_combined(
                        data, p_pred, model,
                        pModel_dimRel_condDense_chFlat=p_pred_dense1,
                        ev_dimRel_condDense_fr_dim_meanvar=ev_dense1,
                        to_plot_params=to_plot_params,
                        to_plot_internals=to_plot_internals,
                        group_dcond_irr=group_dcond_irr,
                        axs=axs,
                    )[0]

                    parad = dict_cache['prd']
                    if parad == 'unibimanual':
                        if dict_cache['bim']:
                            parad_title = 'bimanual'
                        else:
                            parad_title = 'unimanual'
                    elif parad == 'RT':
                        parad_title = 'RT eye'
                    else:
                        raise ValueError()

                    _, p_rt_ch_dat, _, _, ev_cond = data.get_data_by_cond('all')
                    n = p_rt_ch_dat.sum()
                    title_str = 'Subj %s - 2D, %s\n(N=%d' % (
                    dict_cache['sbj'], parad_title, n)
                    if loss_NLL_test is not None:
                        title_str += ', CE=%1.3f, NLL=%1.1f)' % npys(
                            loss_CE_test,
                            loss_NLL_test)
                    else:
                        title_str += ')'

                    plt.suptitle(title_str)
                    dict_fig = {
                        **dict_cache,
                        'grir': to_group_irr,
                        'prms': to_plot_params,
                        'ints': to_plot_internals,
                    }
                    file = locfile.get_file_fig('fit_combined', dict_fig)
                    plt.savefig(file, dpi=150)
                    print('Saved to %s' % file)
                    plt.show()
                    print('--')

        dict_cache = dict_caches[ii_subj, 0]
        data = datas[ii_subj, 0]
        model = models[ii_subj, 0]
        if 'rt_distrib' in to_plot:
            dict_cache['td'] = [v['td'] for v in dict_caches[ii_subj]]
            p_preds_avg_subj = p_preds[ii_subj]

            plot_rt_distrib_pred_data(
                p_preds_avg_subj,
                n_cond_rt_ch=npy(data.get_data_by_cond('all')[1]),
                ev_cond_dim=data.ev_cond_dim,
                dt_model=model.dt, dt_data=data.dt,
            )

            plt.suptitle('Subject %s, eye RT' % dict_cache['sbj'])

            dict_fig = dict_cache
            file = locfile.get_file_fig('rt_distrib_all', dict_fig)
            plt.savefig(file, dpi=300)
            print('Saved to %s' % file)
            plt.show()

            print('losses: ')
            print(losses_NLL_test[ii_subj])
            print('--')

        if 'ch_rt' in to_plot:
            loss_NLL_test = None
            loss_CE_test = None
            # loss.backward()
            to_plot_params = False
            to_plot_internals = False

            axs = plt2.GridAxes(
                3, 2,
                top=0.4, left=0.6, right=0.1, bottom=0.65,
                wspace=0.35, hspace=[0.5, 0.25],
                heights=1.7, widths=2.2
            )

            for i_model, (
                    data, model, p_pred, p_pred_dense1, ev_dense1
            ) in enumerate(zip(
                    datas[ii_subj], models[ii_subj], p_preds[ii_subj],
                    p_pred_dense_subj, ev_dense_subj
                )):

                axs1 = axs[[0, 1 + i_model], :]

                plot_fit_combined(
                    data, npy(p_pred), model,
                    pModel_dimRel_condDense_chFlat=p_pred_dense1,
                    ev_dimRel_condDense_fr_dim_meanvar=ev_dense1,
                    to_plot_params=to_plot_params,
                    to_plot_internals=to_plot_internals,
                    group_dcond_irr=group_dcond_irr,
                    kw_plot_pred=kw_plot_preds[0],
                    kw_plot_data=kw_plot_data,
                    axs=axs1,
                    to_combine_ch_irr_cond=to_combine_ch_irr_cond,
                )

                plt.sca(axs1[-1, 0])
                plt.ylabel('RT (s)\n(%s model)' % model.dtb.kind)
                for col in [0, 1]:
                    plt.sca(axs1[-1, col])
                    # plt2.detach_yaxis(0, 2.5)
                    # plt.yticks([1, 2])
                    if i_model < 1:
                        plt.gca().set_xticklabels([])
                        plt.xlabel('')

            title_str = get_suptitle_str(data, dict_cache, loss_CE_test)
            plt.suptitle(title_str)
            dict_fig = argsutil.kwdefault({
                'grir': to_group_irr,
                'prms': to_plot_params,
                'ints': to_plot_internals,
            }, **dict_cache)
            dict_fig['td'] = [v.dtb.kind for v in models[0]]
            dict_fig1 = summarize_dict_fig(dict_fig)
            file = locfile.get_file_fig('ch_rt', dict_fig1)
            plt.savefig(file, dpi=300)
            print('Saved to %s' % file)
            # plt.show()
            print('--')

    dict_fig = dict_caches[0, 0].copy()
    dict_fig.update({
        'grir': to_group_irr,
        'sbj': '%s-%s' % (dict_caches[0, 0]['sbj'], dict_caches[-1, 0]['sbj']),
        'td': [v.dtb.kind for v in models[0]]
    })

    # === Combine data
    # p_datas0[subj, cond, RT, ch_flat]
    p_datas0 = np.stack([
        npy(v) for v in p_datas[:, 0]
    ])

    # p_data[cond, RT, ch_flat]
    p_data = p_datas0.sum(0)
    p_data = p_data / p_data.sum((1, 2), keepdims=True)
    data = datas[0, 0]  # type: sim2d.Data2DRT

    # p_preds0[model, dim_rel, subj, cond, RT, ch_flat]
    siz0 = p_pred_dense[0].shape
    p_preds0 = np.stack(p_pred_dense)
    p_preds0 = p_preds0.reshape(
        (n_subj, n_dtb2d) + siz0)
    p_preds0 = np.transpose(p_preds0, [1, 2, 0, 3, 4, 5])

    # p_preds0 = np.swapaxes(np.stack([
    #     np.stack(p_preds1) for p_preds1 in p_preds
    # ]), 0, 1)
    # p_preds0 = p_preds0 * p_datas0.sum((2, 3), keepdims=True)[None]

    # p_preds_avg_subj[model, dim_rel, cond, RT, ch_flat]
    p_preds_avg_subj = p_preds0.sum(2)
    p_preds_avg_subj = p_preds_avg_subj / p_preds_avg_subj.sum((3, 4),
                                                               keepdims=True)

    ev_preds0 = deepcopy(ev_dense1)
    ev_max = np.empty([2])
    # if normalize_ev:
    for dim_rel in range(consts.N_DIM):
        ev_max[dim_rel] = np.amax(ev_preds0[dim_rel, :, :, dim_rel, :])
        if normalize_ev:
            ev_preds0[dim_rel] = ev_preds0[dim_rel] / ev_max[dim_rel]
            ev_max[dim_rel] = 1.

    data = get_data_combined(datas, normalize_ev=normalize_ev)

    if 'gof' in to_plot:
        # gof[subj, model]
        gof = losses_NLL_test
        dgof = gof[:, 0] - gof[:, 1]
        max_dgof = np.amax(np.abs(dgof))
        y = np.arange(n_subj)

        axs = plt2.GridAxes(
            1, 1,
            top=0.25, bottom=0.5, left=0.3, right=0.3,
            heights=[n_subj * 0.25],
            widths=[1.],
        )
        plt.sca(axs[0, 0])

        plt.barh(y, dgof, color='k')
        plt.title(get_title_str(parad))

        for x in [-np.log(100), np.log(100)]:
            plt.axvline(x, linewidth=0.25, linestyle='--',
                        color='lightgray')

        plt.xlim(np.array([-1.1, 1.1]) * max_dgof)
        plt.xticks([-max_dgof, 0, max_dgof],
                   [r'$\leftarrow$' + '\nserial', '',
                    r'$\rightarrow$' + '\nparallel'])
        # subjs = consts.SUBJS
        plt2.detach_axis('x', amin=-max_dgof, amax=max_dgof)
        # plt2.detach_axis('y', amin=0, amax=n_subj - 1)
        plt.yticks([])
        plt2.box_off(['top', 'left', 'right'])
        plt.ylabel('participant')

        dict_fig1 = summarize_dict_fig(dict_fig)
        for ext1 in ['.png', '.pdf']:
            file = locfile.get_file_fig('gof', dict_fig1, ext=ext1)
            plt.savefig(file, dpi=300)
            print('Saved to %s' % file)

        print('--')

    if 'rt_distrib_avg' in to_plot:
        n_cond_rt_ch = npy(data.get_data_by_cond('all')[1])
        plot_rt_distrib_pred_data(
            p_preds_avg_subj,
            n_cond_rt_ch=npy(data.get_data_by_cond('all')[1]),
            ev_cond_dim=data.ev_cond_dim,
            dt_model=model.dt, dt_data=data.dt,
        )

        plt.suptitle('All subjects, eye RT')

        file = locfile.get_file_fig('rt_distrib_all', dict_fig)
        plt.savefig(file, dpi=300)
        print('Saved to %s' % file)
        plt.show()

    if 'combined_avg' in to_plot:
        for to_plot_params in [False]:
            for p_pred, model in zip(p_preds_avg_subj, models[0]):
                dict_fig1 = deepcopy(dict_fig)
                dict_fig1.update({
                    'grir': to_group_irr,
                    'td': model.dtb.kind
                })
                dict_fig1 = summarize_dict_fig(dict_fig1)
                plot_fit_combined(
                    data, None, model,
                    pModel_dimRel_condDense_chFlat=p_pred,
                    ev_dimRel_condDense_fr_dim_meanvar=ev_preds0,
                    # dict_fig1,
                    to_combine_ch_irr_cond=to_combine_ch_irr_cond,
                    best_loss=best_loss,
                    to_plot_params=to_plot_params)

    if 'ch_rt_avg' in to_plot:
        loss_CE_test = None
        # loss_CE.backward()
        to_plot_params = False
        to_plot_internals = False

        # WORKING HERE: adjust margins
        axs = plt2.GridAxes(
            3, 2,
            top=0.4, left=0.6, right=0.1, bottom=0.65,
            wspace=0.35, hspace=[0.5, 0.25],
            heights=1.7, widths=2.2
        )

        for i_model, (data, model, p_pred) in enumerate(zip(
                [data] * 2, models[0], p_preds_avg_subj)):

            axs1 = axs[[0, 1 + i_model], :]

            plot_fit_combined(
                data, None, model,
                pModel_dimRel_condDense_chFlat=p_pred,
                ev_dimRel_condDense_fr_dim_meanvar=ev_preds0,
                to_plot_params=to_plot_params,
                to_plot_internals=to_plot_internals,
                group_dcond_irr=group_dcond_irr,
                kw_plot_pred=kw_plot_preds[0],
                kw_plot_pred_ch=kw_plot_preds[i_model],
                kw_plot_data=kw_plot_data,
                to_combine_ch_irr_cond=to_combine_ch_irr_cond,
                axs=axs1,
            )

            plt.sca(axs1[-1, 0])
            plt.ylabel('RT (s)\n(%s model)' % model.dtb.kind)

            # plt2.detach_yaxis(0.5, 2.5)
            # plt.yticks([1, 2])
            # plt.ylim([0.5, 2.5])
            # plt.yticks(np.arange(0.5, 3, 0.5))

            if i_model < 1:
                for col in [0, 1]:
                    for row in [0, 1]:
                        plt.sca(axs1[row, col])
                        plt2.box_off(['bottom'])
                        plt.xlabel('')
                        plt.xticks([])

        plt2.sameaxes(axs[[1, 2], :2], xy='y')
        for row in [1, 2]:
            for col in [0, 1]:
                ax = axs[row, col]
                y_lim = ax.get_ylim()
                x_lim = ax.get_xlim()
                plt2.detach_axis('y', ax=ax,
                                 amin=y_lim[0],
                                 amax=y_lim[1])
                if col == 0:
                    plt2.ticks(ax, 'y', interval=1.)

        for col in [0, 1]:
            plt.sca(axs[-1, col])
            txt = '%s strength' % consts.DIM_NAMES_LONG[col].lower()
            if normalize_ev:
                txt += '\n(a.u.)'
            plt.xlabel(txt)
            xticks = [-ev_max[col], 0, ev_max[col]]
            plt.xticks(xticks, ['%g' % v for v in xticks])

        title_str = get_title_str(parad)
        plt.suptitle(title_str)
        dict_fig = {
            **dict_fig,
            'grir': to_group_irr,
            'prms': to_plot_params,
            'ints': to_plot_internals,
        }
        dict_fig1 = summarize_dict_fig(dict_fig)
        for ext1 in ['.png', '.pdf']:
            file = locfile.get_file_fig('ch_rt_avg', dict_fig1, ext=ext1)
            plt.savefig(file, dpi=300)
            print('Saved to %s' % file)

    # print('--')
    return models, datas


def get_title_str(parad):
    if parad == 'RT':
        title_str = 'Eye'
    # elif parad == 'unibimanual':
    #     if dict_cache['bim']:
    #         title_str = 'Experiment 4 (Bimanual RT)'
    #     else:
    #         title_str = 'Hand'
    elif parad == 'unimanual':
        title_str = 'Hand'
    elif parad == 'bimanual':
        title_str = 'Bimanual'
    else:
        raise ValueError()
        # title_str = parad
    return title_str


def main_fit(
        model: sim2d.FitRT2D = sim2d.FitRT2D,
        # --- Data
        data: sim2d.Data2DRT = None,
        dict_cache: dict = None,
        dict_subdir: dict = None,
        locfile1=None,
        subj: Union[int, str] = None,
        parad='RT',
        # subj=None,
        trial_st=0,
        # bimanual=False,
        # --- Training
        n_fold_valid=1,
        mode_train='easiest',
        ignore_hard_RT=False,
        subsample_factor=5,
        # --- Misc options
        fit_mode='auto',
        kw_name=(),
        ignore_cache=False,
        to_debug=False,
        continue_fit=False,
        save_results=None,
        # subdir='',
        dict_cache_only=False,
        to_compute_grad=True,
        skip_fit_if_absent=False,
        **kwargs
) -> (torch.nn.Module, sim2d.Data2DRT, dict, float, dict):
    """
    :param model:
    :param subj: int or str
    :param parad:
    :param n_fold_valid:
    :param bimanual:
    :param kw_name:
    :param ignore_cache:
    :param subsample_factor:
    :param mode_train:
    :param to_debug:
    :param continue_fit:
    :param trial_st:
    :param kwargs: fed to initializing model and to sim2d.fit_dtb
    :return: model, data, dict_cache, dict_subdir, d
    """

    if type(subj) is not str:
        import warnings
        warnings.warn('Give subj as str, not int!', DeprecationWarning)

    if fit_mode not in ['auto', 'dict_cache_only', 'init_model']:
        raise NotImplementedError()
    # assert fit_mode in ['auto', 'new', 'continue', 'load', 'replicate',
    #                     'dict_cache_only', 'init_model']
    dict_cache_only = fit_mode == 'dict_cache_only'

    assert 'bimanual' not in kwargs, 'use unimanual/bimanual instead of ' \
                                     'unibimanual!'
    if locfile1 is None:
        locfile1 = locfile

    # --- Get data by subj and parad
    if data is None:
        # if subj is None:
        #     subj = consts.SUBJS[parad][i_subj]
        # else:
        #     i_subj = consts.SUBJS[parad].index(subj)
        #
        data, subj, dat = get_data_2D(
            subj, parad,
            # bimanual=bimanual,
            trial_st=trial_st,
            subsample_factor=subsample_factor
        )

    # === Build model
    if type(model) is type:
        model = model(dt=data.dt, nt=data.nt, **kwargs)

    # === Get cache
    kw_name = dict(kw_name)
    if dict_subdir is None:
        dict_subdir = {
            'tnd': model.tnds[0].kind,
            'tndloc': model.tnds[0].loc_kind,
            'tnddsp': model.tnds[0].disper_kind,
            'lps': model.lapse.kind,
            'mdtrn': mode_train,
            'igrt': ignore_hard_RT,
            'sm1': model.to_normalize_within_cond,
            **kw_name
        }

    if dict_cache is None:
        # if parad == 'unibimanual':
        #     kw_name = argsutil.kwdef({
        #         'bim': bimanual
        #     }, kw_name)

        dict_cache = {
            'sbj': subj,
            'prd': parad,
            'ndim': 2,
            'td': model.dtb.kind,
            **dict_subdir
        }
    else:
        dict_cache = dict_cache

    if dict_cache_only:
        return model, data, dict_cache, dict_subdir, None

    if fit_mode == 'init_model':
        return model, dict_cache, dict_subdir

    cache = locfile1.get_cache('fit', dict_cache, subdir=dict_subdir)
    cache_exists = cache.exists()
    if cache_exists and not ignore_cache:
        best_loss, best_state = cache.getdict(['best_loss', 'best_state'])
        loaded_cache = True

        # Always load best_state
        model.load_state_dict(best_state)
    else:
        loaded_cache = True
        best_loss = None

    best_state = model.state_dict()  # missing params filled in by model

    if ((best_loss is None) and (not skip_fit_if_absent)) or continue_fit:
        # == Fit and get best_loss and best_state
        if cache_exists:
            if continue_fit:
                reason = 'User chose to continue from'
            elif ignore_cache:
                reason = 'User chose to ignore'
            else:
                reason = 'For an unspecified reason, ignoring'
            print('%s cache at %s\n= %s'
                  % (reason, cache.fullpath, cache.fullpath_orig))
        print('Fitting model..')
        # # Test run
        # p_rt_ch_pred = model(ev_cond)
        # cost = dtb.fun_loss(p_rt_ch_pred, p_rt_ch_dat)
        # cost.backward()
        # pprint([(v[0], v[1].data, v[1].grad) for v in
        #         model.named_parameters()])

        best_loss, best_state, d, plotfuns = sim2d.fit_dtb(
            model, data,
            n_fold_valid=n_fold_valid,
            mode_train=mode_train,
            to_debug=to_debug,
            comment=argsutil.dict2fname(dict_cache),
            **kwargs
        )
        d1 = {k: v for k, v in d.items() if k.startswith('loss_')}
        d1.update({
            'best_loss': best_loss,
            'best_state': best_state
        })
        cache.set(d1)
        cache.save()

        print('model (fit):')
        print(model.__str__())

    elif skip_fit_if_absent:
        return None, dict_cache, None, None, None, None, None

    else:
        # Otherwise, just get plotfuns
        model.load_state_dict(best_state, strict=False)
        best_state0 = best_state
        # best_state = model.state_dict()
        kw = {
            **kwargs,
            'n_fold_valid': n_fold_valid,
            'mode_train': mode_train,
            'max_epoch': 0,
            'learning_rate': 0,
            'to_debug': to_debug,
            'comment': argsutil.dict2fname(dict_cache),
        }
        best_loss, best_state, d, plotfuns = sim2d.fit_dtb(
            model, data,
            **kw
        )
        for k in best_state.keys():
            dif = npy(best_state[k] - best_state0[k])
            if np.any(np.abs(dif) > 1e-12):
                import warnings
                warnings.warn(
                    'Strange! Difference > 1e-12 for best_state[%s]:' % k)
                print(dif)
                print('--')

    d.update({
        'best_loss': best_loss,
        'best_state': best_state
    })

    print('model (fit):')
    print(model.__str__())

    # == Compute gradient for plotting
    model.load_state_dict(best_state)
    ev_cond_fr_dim_meanvar, n_cond_rt_ch = data.get_data_by_cond('all')[:2]

    if to_compute_grad:
        p_cond__rt_ch_pred = model(ev_cond_fr_dim_meanvar)
        cost = sim2d.fun_loss(p_cond__rt_ch_pred, n_cond_rt_ch)
        cost.backward()

    if save_results is None:
        save_results = not (loaded_cache or skip_fit_if_absent)
    if save_results:
        # save_cache(cache, d, best_loss, best_state, subdir)

        save_fit_results(model, best_state, d, plotfuns,
                         locfile1, dict_cache, dict_subdir)
        # if subdir is not None:
        #     for file in files:
        #         localfile.copy2subdir(file, subdir)

    pprint([(v[0], v[1].data, v[1].grad) for v in
            model.named_parameters()])

    return model, data, dict_cache, dict_subdir, d


def main_plot_fit_from_matlab(
        to_group_irr=True,
        to_plot=('ch_rt', 'rt'),
        to_plot_scale=False,
        to_cumsum=False,
):

    from data_2d import load_matlab_fit as mfit
    dat_fit = deepcopy(mfit.dat)
    # p_data = dat_fit['data_avg_subj']
    p_pred = dat_fit['pred_avg_subj']  # type: np.ndarray
    nt = p_pred.shape[0]
    n_conds = np.prod(p_pred.shape[1:3])
    n_chs = np.prod(p_pred.shape[3:5])

    # p_preds_avg_subj[model, cond_flat, RT, ch_flat]
    p_preds_avg_subj = np.transpose(
        p_pred.reshape([nt, n_conds, n_chs, -1]),
        [3, 1, 0, 2]
    )

    outs = [
        get_data_2D(
            i_subj, parad='RT', bimanual=False,
            trial_st=200,
            subsample_factor=1
        )[:2] for i_subj in [1, 2, 3]
    ]
    datas = np.array([v[0] for v in outs], dtype=np.object)
    subjs = [v[1] for v in outs]
    data = get_data_combined(datas)
    model = sim2d.FitRT2D(dt=data.dt, nt=data.nt)

    dict_fig = {
        'prd': 'RT',
        'sbj': subjs,
        'dt': ['serial', 'parallel'],
        'fit': 'matlab'
    }

    if 'ch_rt' in to_plot:
        loss_CE = None
        to_plot_params = False
        to_plot_internals = False

        axs = None
        rts0 = []
        for p_pred, kw_plot_pred in zip(p_preds_avg_subj, kw_plot_preds):
            axs, rts1 = plot_fit_combined(
                data, npy(p_pred), model,
                to_plot_params=to_plot_params,
                to_plot_internals=to_plot_internals,
                to_combine_ch_irr_cond=True,
                # group_dcond_irr=group_dcond_irr,
                kw_plot_pred=kw_plot_pred,
                kw_plot_data=kw_plot_data,
                axs=axs,
            )[:2]
            rts0.append(rts1)
        for dim_rel in [0, 1]:
            axs[1, dim_rel].set_xticks([-1, 0, 1])
            axs[1, dim_rel].set_xticklabels(consts.XTICKLABELS[dim_rel])

        rts = np.array(rts0)
        drts = rts[1, 1] - rts[0, 1]
        print(drts)
        drt = np.amax(np.abs(drts.flatten()))
        idrt = np.argmax(np.abs(drts.flatten()))
        rt_range = rts.reshape([2, 2, -1])[:, 1, idrt]
        print((drt, rt_range))

        if to_plot_scale:
            x = np.zeros(2) + 0.
            axs[1, 1].plot(x, rt_range, 'k-', linewidth=0.5)
            axs[1, 1].text(x[0] - 0.21, np.mean(rt_range),
                           '%1.0f ms' % (drt * 1e3),
                           ha='right', va='center')

        title_str = get_suptitle_str(data, dict_fig, to_show_n=False)
        plt.suptitle(title_str)
        dict_fig1 = argsutil.kwdefault({
            'prms': to_plot_params,
            'ints': to_plot_internals,
            'scl': to_plot_scale,
        }, **dict_fig)

        for ext in ['.png', '.pdf']:
            file = locfile.get_file_fig('ch_rt', dict_fig1, ext=ext)
            plt.savefig(file, dpi=300)
            print('Saved to %s' % file)
        # plt.show()
        print('--')

    if 'rt_distrib' in to_plot:
        plot_rt_distrib_pred_data(
            p_preds_avg_subj[:, :, :-1, :],
            n_cond_rt_ch=npy(data.get_data_by_cond('all')[1]),
            ev_cond_dim=data.ev_cond_dim,
            dt_model=model.dt, dt_data=data.dt,
            smooth_sigma_sec=0.1,
            to_cumsum=to_cumsum,
            to_plot_scale=to_plot_scale
        )
        dict_fig1 = deepcopy(dict_fig)
        dict_fig1.update({'csm': to_cumsum})

        plt.suptitle('All subjects, eye RT')

        for ext in ['.png', '.pdf']:
            file = locfile.get_file_fig('rt_distrib_all', dict_fig1, ext=ext)
            plt.savefig(file, dpi=300)
            print('Saved to %s' % file)
        # plt.show()
        print('--')


def get_data_2D(i_subj: Union[str, int] = 0, parad='RT',
                bimanual: bool = None,
                trial_st=0,
                subsample_factor=1,
                ) -> (dict, np.ndarray, np.ndarray, np.ndarray, str):
    """
    :param i_subj:
    :param parad:
    :return: dat, ch_by_dim[tr, dim], rt[tr], cond_by_dim[tr, dim], subj
    """

    if parad == 'unimanual':
        assert (bimanual is None) or (not bimanual)
        bimanual = False
        parad = 'unibimanual'

    elif parad == 'bimanual':
        assert (bimanual is None) or bimanual
        bimanual = True
        parad = 'unibimanual'

    # Choose by dim_rel and parad
    dat0 = load_data.load_data_combined()
    dat = np2.filt_dict(dat0, (
        np.all(dat0['dim_rel'], 1)
        & (dat0['id_parad'] == dat0['parads'].index(parad))
    ))

    if parad == 'unibimanual':
        if bimanual is not None:
            dat = np2.filt_dict(dat, 
                                dat['bimanual'].astype(np.bool) == bimanual)



    # if parad == 'unibimanual':

    # Choose subject
    id_subjs = np.unique(dat['id_subj'])
    if type(i_subj) is str:
        subj = i_subj
        id_subj = dat['subjs'].index(subj)
        i_subj = list(id_subjs).index(id_subj)
    else:
        id_subj = id_subjs[i_subj]
        subj = dat['subjs'][id_subj]
    dat = np2.filt_dict(dat, dat['id_subj'] == id_subj)

    print('Subject: %s (%d/%d) out of: ' %
          (subj, i_subj, len(id_subjs)), end='')
    print(np.array(dat['subjs'])[id_subjs]) # to check

    # ev and ch
    incl = np.all(~np.isnan(dat['ch']), 1) & ~np.isnan(dat['RT'])
    ch_tr_dim = (dat['ch'][incl, :] - 1).astype(np.long)  # [tr, dim]
    rt_tr = dat['RT'][incl]  # type: np.ndarray
    ev_tr_dim = dat['cond'][incl, :]  # [tr, dim]

    ch_tr_dim = ch_tr_dim[trial_st:]
    rt_tr = rt_tr[trial_st:]
    ev_tr_dim = ev_tr_dim[trial_st:]

    data = sim2d.Data2DRT(
        ev_tr_dim, ch_tr_dim, rt_tr,
        subsample_factor=subsample_factor
    )
    return data, subj, dat


def plot_fit_combined(
        data: Union[sim2d.Data2DRT, dict] = None,
        pModel_cond_rt_chFlat=None, model=None,
        pModel_dimRel_condDense_chFlat=None,
        # --- in place of data:
        pAll_cond_rt_chFlat=None,
        evAll_cond_dim=None,
        pTrain_cond_rt_chFlat=None,
        evTrain_cond_dim=None,
        pTest_cond_rt_chFlat=None,
        evTest_cond_dim=None,
        dt=None,
        # --- optional
        ev_dimRel_condDense_fr_dim_meanvar=None,
        dt_model=None,
        to_plot_internals=True,
        to_plot_params=True,
        to_plot_choice=True,
        # to_group_irr=False,
        group_dcond_irr=None,
        to_combine_ch_irr_cond=True,
        kw_plot_pred=(),
        kw_plot_pred_ch=(),
        kw_plot_data=(),
        axs=None,
):
    """

    :param data:
    :param pModel_cond_rt_chFlat:
    :param model:
    :param pModel_dimRel_condDense_chFlat:
    :param ev_dimRel_condDense_fr_dim_meanvar:
    :param to_plot_internals:
    :param to_plot_params:
    :param to_group_irr:
    :param to_combine_ch_irr_cond:
    :param kw_plot_pred:
    :param kw_plot_data:
    :param axs:
    :return:
    """
    # ev_cond[cond, dim]
    # if isinstance(data, dict):
    #     pAll_cond_rt_chFlat = data['pAll_cond_rt_chFlat']
    #     evAll_cond_dim = data['evAll_cond_dim']
    #     pTrain_cond_rt_chFlat = data['pTrain_cond_rt_chFlat']
    #     evTrain_cond_dim = data['evTrain_cond_dim']
    #     pTest_cond_rt_chFlat = data['pTest_cond_rt_chFlat']
    #     evTest_cond_dim = data['evTest_cond_dim']
    #     dt = data['dt']
    # else:
    if data is None:
        if pTrain_cond_rt_chFlat is None:
            pTrain_cond_rt_chFlat = pAll_cond_rt_chFlat
        if evTrain_cond_dim is None:
            evTrain_cond_dim = evAll_cond_dim
        if pTest_cond_rt_chFlat is None:
            pTest_cond_rt_chFlat = pAll_cond_rt_chFlat
        if evTest_cond_dim is None:
            evTest_cond_dim = evAll_cond_dim
    else:
        _, pAll_cond_rt_chFlat, _, _, evAll_cond_dim = \
            data.get_data_by_cond('all')
        _, pTrain_cond_rt_chFlat, _, _, evTrain_cond_dim = data.get_data_by_cond(
            'train_valid', mode_train='easiest')
        _, pTest_cond_rt_chFlat, _, _, evTest_cond_dim = data.get_data_by_cond(
            'test', mode_train='easiest')
        dt = data.dt
    hs = {}

    if model is None:
        assert not to_plot_internals
        assert not to_plot_params

    if dt_model is None:
        if model is None:
            dt_model = dt
        else:
            dt_model = model.dt

    if axs is None:
        if to_plot_params:
            axs = plt2.GridAxes(3, 3)  # TODO: beautify ratios
            # fig = plt.figure(figsize=[6 * 1.4, 6 * 5.3 / 4.3])
            # gs = plt.GridSpec(
            #     nrows=6, ncols=5,
            #     bottom=0.08, top=0.91,
            #     left=0.12, right=0.99,
            #     wspace=0., hspace=0.2,
            #     height_ratios=[1, 1, 0.3, 1, 1, 1],
            #     width_ratios=[1, 0.5, 1, 2., 1.5]
            # )
        else:
            if to_plot_internals:
                axs = plt2.GridAxes(3, 3)  # TODO: beautify ratios
                # fig = plt.figure(figsize=[6 * 0.7, 6 * 5.3 / 4.3])
                # gs = plt.GridSpec(
                #     nrows=6, ncols=3,
                #     bottom=0.08, top=0.91,
                #     left=0.2, right=0.99,
                #     wspace=0., hspace=0.2,
                #     height_ratios=[1, 1, 0.3, 1, 1, 1],
                #     width_ratios=[1, 0.5, 1]
                # )
            else:
                if to_plot_choice:
                    axs = plt2.GridAxes(2, 2)  # TODO: beautify ratios
                    # fig = plt.figure(figsize=[6 * 0.7, 6 * 2.3 / 4.3])
                    # gs = plt.GridSpec(
                    #     nrows=3, ncols=3,
                    #     bottom=0.08, top=0.9,
                    #     left=0.14, right=0.98,
                    #     wspace=0., hspace=0.2,
                    #     height_ratios=[1, 1, 0.15],
                    #     width_ratios=[1, 0.3, 1]
                    # )
                else:
                    axs = plt2.GridAxes(1, 2)  # TODO: beautify ratios

    rts = []
    hs['rt'] = []
    for dim_rel in range(consts.N_DIM):
        # --- data_pred may not have all conditions, so concatenate the rest
        #  of the conditions so that the color scale is correct. Then also
        #  concatenate p_rt_ch_data_pred1 with zeros so that nothing is
        #  plotted in the concatenated.
        evTest_cond_dim1 = np.concatenate([
            evTest_cond_dim, evAll_cond_dim
        ], axis=0)
        pTest_cond_rt_chFlat1 = np.concatenate([
            pTest_cond_rt_chFlat, np.zeros_like(pAll_cond_rt_chFlat)
        ], axis=0)

        if ev_dimRel_condDense_fr_dim_meanvar is None:
            evModel_cond_dim = evAll_cond_dim
        else:
            if ev_dimRel_condDense_fr_dim_meanvar.ndim == 5:
                evModel_cond_dim = npy(ev_dimRel_condDense_fr_dim_meanvar[
                                           dim_rel][:, 0, :, 0])
            else:
                assert ev_dimRel_condDense_fr_dim_meanvar.ndim == 4
                evModel_cond_dim = npy(ev_dimRel_condDense_fr_dim_meanvar[
                                           dim_rel][:, 0, :])
            pModel_cond_rt_chFlat = npy(pModel_dimRel_condDense_chFlat[dim_rel])

        if to_plot_choice:
            # --- Plot choice
            ax = axs[0, dim_rel]
            plt.sca(ax)

            if to_combine_ch_irr_cond:
                ev_cond_model1, p_rt_ch_model1 = combine_irr_cond(
                    dim_rel, evModel_cond_dim, pModel_cond_rt_chFlat
                )

                sim2d.plot_p_ch_vs_ev(ev_cond_model1, p_rt_ch_model1,
                                      dim_rel=dim_rel, style='pred',
                                      group_dcond_irr=None,
                                      kw_plot=kw_plot_pred_ch,
                                      cmap=lambda n: lambda v: [0., 0., 0.],
                                      )
            else:
                sim2d.plot_p_ch_vs_ev(evModel_cond_dim, pModel_cond_rt_chFlat,
                                      dim_rel=dim_rel, style='pred',
                                      group_dcond_irr=group_dcond_irr,
                                      kw_plot=kw_plot_pred,
                                      cmap=cmaps[dim_rel]
                                      )
            hs, conds_irr = sim2d.plot_p_ch_vs_ev(
                evTest_cond_dim1, pTest_cond_rt_chFlat1,
                dim_rel=dim_rel, style='data_pred',
                group_dcond_irr=group_dcond_irr,
                cmap=cmaps[dim_rel],
                kw_plot=kw_plot_data,
            )
            hs1 = [h[0] for h in hs]
            odim = 1 - dim_rel
            odim_name = consts.DIM_NAMES_LONG[odim]
            legend_odim(conds_irr, hs1, odim_name)
            sim2d.plot_p_ch_vs_ev(evTrain_cond_dim, pTrain_cond_rt_chFlat,
                                  dim_rel=dim_rel, style='data_fit',
                                  group_dcond_irr=group_dcond_irr,
                                  cmap=cmaps[dim_rel],
                                  kw_plot=kw_plot_data
                                  )
            plt2.detach_axis('x', np.amin(evTrain_cond_dim[:, dim_rel]),
                             np.amax(evTrain_cond_dim[:, dim_rel]))
            ax.set_xlabel('')
            ax.set_xticklabels([])
            if dim_rel != 0:
                plt2.box_off(['left'])
                plt.yticks([])

            ax.set_ylabel('P(%s choice)' % consts.CH_NAMES[dim_rel][1])

        # --- Plot RT
        ax = axs[int(to_plot_choice) + 0, dim_rel]
        plt.sca(ax)
        hs1, rts1 = sim2d.plot_rt_vs_ev(
            evModel_cond_dim,
            pModel_cond_rt_chFlat,
            dim_rel=dim_rel, style='pred',
            group_dcond_irr=group_dcond_irr,
            dt=dt_model,
            kw_plot=kw_plot_pred,
            cmap=cmaps[dim_rel]
        )
        hs['rt'].append(hs1)
        rts.append(rts1)

        sim2d.plot_rt_vs_ev(evTest_cond_dim1, pTest_cond_rt_chFlat1,
                            dim_rel=dim_rel, style='data_pred',
                            group_dcond_irr=group_dcond_irr,
                            dt=dt,
                            cmap=cmaps[dim_rel],
                            kw_plot=kw_plot_data
                            )
        sim2d.plot_rt_vs_ev(evTrain_cond_dim, pTrain_cond_rt_chFlat,
                            dim_rel=dim_rel, style='data_fit',
                            group_dcond_irr=group_dcond_irr,
                            dt=dt,
                            cmap=cmaps[dim_rel],
                            kw_plot=kw_plot_data
                            )
        plt2.detach_axis('x', np.amin(evTrain_cond_dim[:, dim_rel]),
                         np.amax(evTrain_cond_dim[:, dim_rel]))
        if dim_rel != 0:
            ax.set_ylabel('')
            plt2.box_off(['left'])
            plt.yticks([])

        ax.set_xlabel(consts.DIM_NAMES_LONG[dim_rel].lower() + ' strength')

        if dim_rel == 0:
            ax.set_ylabel('RT (s)')

        if to_plot_internals:
            for ch1 in range(consts.N_CH):
                ch0 = dim_rel
                ax = axs[3 + ch1, dim_rel]
                plt.sca(ax)

                ch_flat = consts.ch_by_dim2ch_flat(np.array([ch0, ch1]))
                model.tnds[ch_flat].plot_p_tnd()
                ax.set_xlabel('')
                ax.set_xticklabels([])
                ax.set_yticks([0, 1])
                if ch0 > 0:
                    ax.set_yticklabels([])

                ax.set_ylabel(r"$\mathrm{P}(T^\mathrm{n} \mid"
                              " \mathbf{z}=[%d,%d])$"
                              % (ch0, ch1))

            ax = axs[5, dim_rel]
            plt.sca(ax)
            if hasattr(model.dtb, 'dtb1ds'):
                model.dtb.dtb1ds[dim_rel].plot_bound(color='k')

    plt2.sameaxes(axs[-1, :consts.N_DIM], xy='y')

    if to_plot_params:
        ax = axs[0, -1]
        plt.sca(ax)
        model.plot_params()

    return axs, rts, hs


def legend_odim(conds_irr, hs1, odim_name, **kwargs):
    return plt.legend(
        hs1, ['%g' % v for v in conds_irr],
        title=r'$\left|\mathrm{%s~str}\right|$' % odim_name, **{
            'loc': 'lower right',
            'handlelength': 0.5,
            'handletextpad': 0.35,
            'labelspacing': 0.15,
            'borderpad': 0.,
            'bbox_to_anchor': [0., -0.02, 1., 1.],
            'frameon': False,
            **kwargs
        }
    )


def combine_irr_cond(dim_rel, evAll_cond_dim, pAll_cond_rt_chFlat):
    conds_rel = np.unique(evAll_cond_dim[:, dim_rel])
    n_conds_rel = len(conds_rel)
    p_rt_ch_dat1 = np.zeros((n_conds_rel,) + pAll_cond_rt_chFlat.shape[1:])
    ev_cond1 = np.zeros((n_conds_rel,) + evAll_cond_dim.shape[1:])
    for i_cond, cond in enumerate(conds_rel):
        incl = evAll_cond_dim[:, dim_rel] == cond
        p_rt_ch_dat1[i_cond] = pAll_cond_rt_chFlat[incl].sum(0)
        ev_cond1[i_cond, dim_rel] = cond
    return ev_cond1, p_rt_ch_dat1


def summarize_dict_fig(dict_fig):
    return {
        k: v
        for k, v in dict_fig.items()
        if k in ['lps', 'mdtrn', 'prd', 'sbj', 'td', 'tnd', 'grir']
    }


def get_data_combined(datas, normalize_ev=True):
    data0 = deepcopy(datas.flatten()[0])  # type: sim2d.Data2DRT
    evs = []
    chs = []
    rts = []

    def normalize(v):
        return v / np.amax(v, 0, keepdims=True)

    for data in datas.flatten():
        ev1 = deepcopy(data.ev_tr_dim)
        if normalize_ev:
            ev1 = normalize(ev1)
        evs.append(ev1)
        chs.append(data.ch_tr_dim)
        rts.append(data.rt_tr)
    data = sim2d.Data2DRT(
        np.concatenate(evs, axis=0),
        np.concatenate(chs, axis=0),
        np.concatenate(rts, axis=0),
        subsample_factor=data0.subsample_factor
    )
    return data


def get_suptitle_str(data, dict_cache, loss_CE=None,
                     to_show_n=True):
    """

    :param data:
    :param dict_cache: keys 'prd' (paradigm), 'sbj' (subj)
    :param loss_CE:
    :return:
    """
    parad = dict_cache['prd']
    # if parad == 'unibimanual':
    #     if dict_cache['bim']:
    #         parad_title = 'bimanual'
    #     else:
    #         parad_title = 'unimanual'
    if parad == 'RT':
        parad_title = 'RT eye'
    else:
        parad_title = parad
        # raise ValueError()
    _, p_rt_ch_dat, _, _, ev_cond = data.get_data_by_cond('all')
    n = p_rt_ch_dat.sum()

    subj = dict_cache['sbj']
    # if type(subj) is not str:
    #     subj = subj[0] + '-' + subj[-1]

    title_str = 'Subj %s - 2D, %s' % (
        subj, parad_title)
    if to_show_n or loss_CE is not None:
        title_str += '\n'
    if to_show_n:
        title_str += 'N=%d' % n
    if loss_CE is not None:
        if to_show_n:
            title_str += ', '
        title_str += 'CE=%1.3f, NLL=%1.1f' % npys(
            loss_CE,
            loss_CE * n * np.prod(np.array(p_rt_ch_dat.shape[1:])))
    return title_str


def plot_rt_distrib_pred_data(
        p_pred_cond_rt_ch,
        n_cond_rt_ch, ev_cond_dim, dt_model, dt_data=None,
        smooth_sigma_sec=0.1,
        to_plot_scale=False,
        to_cumsum=False,
        to_normalize_max=True,
        xlim=None,
        colors=('magenta', 'cyan'),
        kw_plot_pred=(),
        kw_plot_data=(),
        to_skip_zero_trials=False,
        labels=None,
        **kwargs
):
    """

    :param n_cond_rt_ch: [cond, rt, ch] = n_tr(cond, rt, ch)
    :param p_pred_cond_rt_ch: [model, cond, rt, ch] = P(rt, ch | cond, model)
    :param ev_cond_dim:
    :param dt_model:
    :param dt_data:
    :param smooth_sigma_sec:
    :param to_plot_scale:
    :param to_cumsum:
    :param xlim:
    :param kwargs:
    :return:
    """

    axs = None
    ps = []
    ps0 = []
    hss = []

    p_pred_cond_rt_ch = p_pred_cond_rt_ch / np.sum(
        p_pred_cond_rt_ch, (-1, -2), keepdims=True)
    n_preds1 = p_pred_cond_rt_ch * np.sum(
        n_cond_rt_ch, (-1, -2))[None, :, None, None]
    nt = p_pred_cond_rt_ch.shape[-2]
    if dt_data is None:
        dt_data = dt_model
    if labels is None:
        labels = [''] * (len(n_preds1) + 1)

    for i_pred, n_pred in enumerate(n_preds1):
        color = colors[i_pred]
        axs, p0, p1, hs = sim2d.plot_rt_distrib(
            n_pred, ev_cond_dim,
            dt=dt_model,
            axs=axs,
            alpha=1.,
            smooth_sigma_sec=smooth_sigma_sec,
            to_skip_zero_trials=to_skip_zero_trials,
            colors=color,
            alpha_face=0,
            to_normalize_max=to_normalize_max,
            to_cumsum=to_cumsum,
            to_use_sameaxes=False,
            kw_plot={
                'linewidth': 1.5,
                **dict(kw_plot_pred),
            },
            label=labels[i_pred],
            **kwargs,
        )[:4]
        ps.append(p1)
        ps0.append(p0)
        hss.append(hs)

    axs, p0, p1, hs = sim2d.plot_rt_distrib(
        n_cond_rt_ch, ev_cond_dim,
        dt=dt_data,
        axs=axs,
        smooth_sigma_sec=smooth_sigma_sec,
        colors='k',
        alpha_face=0.,
        to_normalize_max=to_normalize_max,  # normalize across preds and data instead
        # to_exclude_bins_wo_trials=10,
        to_cumsum=to_cumsum,
        # to_use_sameaxes=False,
        to_skip_zero_trials=to_skip_zero_trials,
        kw_plot={
            'linewidth': 0.5,
            **dict(kw_plot_data),
        },
        label=labels[-1],
        **kwargs,
    )
    ps.append(p1)
    ps0.append(p0)
    hss.append(hs)

    ps = np.stack(ps)
    ps0 = np.stack(ps0)

    ps_flat = np.swapaxes(ps, 0, 2).reshape([ps.shape[1] * ps.shape[2], -1])

    # if to_normalize_max:
    #     # normalize across preds and data within each pane
    #     for ax, ps1 in zip(axs.flatten(), ps_flat):
    #         ymax = np.amax(np.abs(ps1))
    #         # ymin = np.amin(ps1)
    #         ax.set_ylim(bottom=ymax * -1.05, top=ymax * 1.05)
    # else:
    #     ymax = np.amax(ps)
    #     ymin = np.amin(ps)
    #     for ax in axs.flatten():
    #         ax.set_ylim(bottom=ymin * 1.05, top=ymax * 1.05)

    for ax in axs.flatten():
        if xlim is None:
            if to_cumsum:
                xlim = [0.5, 4.5]
            else:
                xlim = [0.5, 4.5]

        plt2.detach_axis('x', *xlim, ax=ax)
        ax.set_xlim(xlim[0] - 0.1, xlim[1] + 0.1)
        # ax.set_ylim(bottom=-0.9, top=1.05)
        #
        # ax.set_xticklabels([])
    axs[-1, 0].set_xticks(xlim)
    axs[-1, 0].set_xticklabels(['%g' % v for v in xlim])

    from lib.pylabyk import numpytorch as npt
    t = torch.arange(nt) * dt_model

    mean_rts = []
    # plt.figure()
    for p1 in ps0:
        p11 = npt.sumto1(torch.tensor(p1).sum([-1, -2])[0, 0, :])
        mean_rts.append(npy((torch.tensor(t) * p11).sum()))
    #     plt.plot(*npys(t, p11))
    #
    # plt.show()
    print('mean_rts:')
    print(mean_rts)
    print(mean_rts[1] - mean_rts[0])

    conds = [np.unique(ev_cond_dim[:, i]) for i in [0, 1]]
    p_preds = torch.tensor(n_preds1).reshape([
        2, len(conds[0]), len(conds[1]), nt, 2, 2
    ]) + 1e-12

    # p_preds12 = npy(p_preds[:, 3:6, 4, :, :, :].sum((-1, -2, -4)))
    # m_preds12 = (npy(t)[None, :] * np2.sumto1(p_preds12, 1)).sum(1)
    # dm_preds12 = m_preds12[1] - m_preds12[0]
    # print('m_preds12 (pooled irr_difs as in mean plots)')
    # print((m_preds12, dm_preds12))

    # middle = np.floor(np.array(p_preds.shape[1:3]) / 2).astype(np.int)
    # p_preds11 = p_preds[:, middle[0], middle[1], :, :, :]
    # m_preds = (t[None, :, None, None] * npt.sumto1(p_preds11, 1)).sum([1])
    # dm_preds = m_preds[1] - m_preds[0]
    # print(m_preds)
    # print(dm_preds)

    # p_preds2 = p_preds11.sum([-1, -2])
    # m_preds2 = (t[None, :] * npt.sumto1(p_preds2, -1)).sum(-1)
    # dm_preds2 = m_preds2[1] - m_preds2[0]
    # print('m_preds2')
    # print(m_preds2)
    # print(dm_preds2)

    # mean_rts = np.array([
    #     npy(npt.mean_distrib(
    #         torch.tensor(t),
    #         npt.sumto1(torch.tensor(p1).sum([-1, -2])[-1, -1, :])))
    #     for p1 in ps])

    if to_plot_scale:
        y = 0.8
        axs[-1, -1].plot(mean_rts[:2], y + np.zeros(2), 'k-', linewidth=0.5)
        x = np.mean(mean_rts[:2])
        plt.text(x, y + 0.1,
                 '%1.0f ms' % (np.abs(mean_rts[1] - mean_rts[0]) * 1e3),
                 ha='center', va='bottom')

    # CHECKED
    # plt.show()
    print('--')

    return axs, hss


def ____Main____():
    pass


if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.double)

    # main_plot_fit_from_matlab(
    #     to_plot=['ch_rt', 'rt_distrib'],
    #     to_cumsum=False,
    #     to_plot_scale=False,
    # )

    # i_subjs = [1, 2, 3]
    # # # i_subjs = [1]  # , 2, 3]
    # parads = ['RT']
    # normalize_ev = True

    i_subjs = np.arange(8)
    # i_subjs = [2, 3]
    # parads = ['unimanual', 'bimanual']
    parads = ['unimanual']
    # parads = ['bimanual']
    normalize_ev = False

    dtb2ds = [sim2d.Dtb2DRTSerial, sim2d.Dtb2DRTParallel]
    # dtb2ds = [sim2d.Dtb2DRTSerial]
    # dtb2ds = [sim2d.Dtb2DRTParallel]

    # for i_subj in i_subjs:
    for parad in parads:
        for to_group_irr in [False]: # [False, True]:
            models, datas = main_fit_subjs(
                subjs=i_subjs,
                # i_subjs=[i_subj],
                dtb2ds=dtb2ds,
                parad=parad,
                to_group_irr=to_group_irr,
                # ['rt_distrib', 'combined', 'ch_rt'],
                # to_plot=['ch_rt', 'ch_rt_avg'],
                mode_train='all',
                # mode_train='easiest',
                # to_plot=['gof'],
                # to_plot=['ch_rt'],
                to_plot=['ch_rt_avg', 'gof'],
                kw_fit={
                    'to_allow_irr_ixn': True
                },
                to_combine_ch_irr_cond=False,
                normalize_ev=normalize_ev
            )

    # main_plot_fits(i_subjs, dtbs, models, datas)