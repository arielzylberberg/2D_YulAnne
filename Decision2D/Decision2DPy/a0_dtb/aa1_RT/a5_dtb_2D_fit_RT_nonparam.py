#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.


import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
from copy import deepcopy
from importlib import reload

import torch

from lib.pylabyk.numpytorch import npy, npys
from lib.pylabyk import localfile, argsutil, np2, plt2, yktorch as ykt

from data_2d import consts, load_data
from a0_dtb import a1_dtb_1D_sim as sim1d, a3_dtb_2D_sim as sim2d
from a0_dtb.aa1_RT import a4_dtb_2D_fit_RT as fit2d
from a0_dtb.aa1_RT.a4_dtb_2D_fit_RT import get_data_2D

locfile = localfile.LocalFile(
    pth_root='../../Data_2D/Data_2D_Py/a0_dtb/RTNonparam',
    cache_dir=''
)


def main(
        n_fold_test=1,
):
    ds = []
    dtb2ds = [sim2d.RTNonparam2DSer, sim2d.RTNonparam2DPar]

    # subjs = ['S1', 'S2', 'S3']
    subjs = ['S1', 'S3']
    for subj in subjs:
        for i_fold_test in range(n_fold_test):
            for dtb2d in dtb2ds:
            # for dtb2d in [sim2d.RTNonparam2DPar]:
                model, data, dict_cache, dict_subdir, d11 = main_fit(
                    subj=subj,
                    dtb2d=dtb2d,
                    n_fold_test=n_fold_test,
                    i_fold_test=i_fold_test,
                    # to_save_res=True,
                    # thres_n_tr=10,
                    # trial_st=200,
                )
                ds.append({
                    'subj': subj,
                    'td': dtb2d.kind,
                    'd': d11,
                    'ifl': i_fold_test,
                    'nfl': n_fold_test,
                    'data': data,
                    'model': model,
                    'loss_NLL': npy(d11['loss_NLL_test']),
                    'loss_BIC': npy(d11['loss_BIC_test']),
                    'out_all': npy(d11['out_all']),
                    'dict_cache': dict_cache
                })

    ds = np2.listdict2dictlist(ds)
    for k in ['loss_NLL', 'loss_BIC']:
        ds[k] = np.array([npy(v) for v in ds[k]]).flatten()
    ds['out_all'] = np.array(ds['out_all'])

    # p_preds = np.stack(
    #     ds['']
    # )
    for subj in subjs:
        incl = np.array([v == subj for v in ds['subj']])
        incl1 = np.nonzero(incl)[0][0]
        data = ds['data'][incl1]
        model = ds['model'][incl1]
        dict_cache = ds['dict_cache'][incl1]

        fit2d.plot_rt_distrib_pred_data(
            ds['out_all'][incl, :],
            n_cond_rt_ch=npy(data.get_data_by_cond('all')[1]),
            ev_cond_dim=data.ev_cond_dim,
            dt_model=model.dt, dt_data=data.dt,
            to_cumsum=True,
            xlim=[0, 5]
        )
        kinds = [dtb2d.kind for dtb2d in dtb2ds]
        dict_fig = {**dict_cache,
                    'td': ('[%s]' % (','.join(kinds)))}
        file = locfile.get_file_fig('rtdstr', dict_fig,
                                    subdir=dict_subdir)
        plt.savefig(file, dpi=72)
        print('Saved to %s' % file)

    print('--')


def main_fit(
        model=None,
        data=None,
        dtb2d=sim2d.RTNonparam2DSer,
        dict_subdir=None,
        dict_cache=None,
        locfile1=None,
        subj='S1',
        parad='RT',
        bimanual=False,
        subsample_factor=5,
        mode_train='easiest',
        n_fold_test=1,
        i_fold_test=0,
        to_debug=False,
        disper_ub=3.,
        lapse_max=1e-6,
        to_save_res=None,
        fit_mode='auto',
        # what trials to include
        correct_only=False,
        exclude_0coh=False,
        trial_st=0,
        thres_n_tr=10,
        sumto1_wi_cond=True,  # CHECKED
        **kwargs
):
    """

    :param model:
    :param data:
    :param dtb2d:
    :param dict_subdir:
    :param dict_cache:
    :param locfile1:
    :param subj:
    :param parad:
    :param bimanual:
    :param subsample_factor:
    :param mode_train:
    :param n_fold_test:
    :param i_fold_test:
    :param to_debug:
    :param disper_ub:
    :param lapse_max:
    :param to_save_res:
    :param fit_mode: 'auto'|'d_only'|'init_model'
    :param correct_only:
    :param exclude_0coh:
    :param trial_st:
    :param thres_n_tr:
    :param sumto1_wi_cond:
    :param kwargs:
    :return: model, data, dict_cache, dict_subdir, d
    """

    if locfile1 is None:
        locfile1 = locfile

    # --- Get data by subj and parad
    if data is None:
        data, subj, _ = get_data_2D(
            subj, parad,
            bimanual=bimanual,
            trial_st=trial_st,
            subsample_factor=subsample_factor
        )

    ev, n_cond_rt_ch, _, _, ev_cond_dim = data.get_data_by_cond(
        'all', mode_train='all')

    if model is None:
        if type(dtb2d) is type:
            dtb2d = dtb2d(
                ev_cond_dim=ev_cond_dim,
                # n_cond_ch=npy(n_cond_rt_ch).sum(1),
                n_cond_rt_ch=npy(n_cond_rt_ch),
                dt=data.dt, nt=data.nt,
                disper_ub=disper_ub,
                thres_n_tr=thres_n_tr,
                sumto1_wi_cond=sumto1_wi_cond,
                correct_only=correct_only,
                exclude_0coh=exclude_0coh,
                **kwargs
            )
        model = sim2d.FitRT2D(
            dtb2d=dtb2d, timer=dtb2d.timer, lapse_max=lapse_max,
            lapse=sim1d.LapseUniformRT  # since choice shouldn't change
        )
    # model.load_state_dict({
    #     'lapse.p_lapse._data': torch.tensor([1e-6])
    # })
    # model.lapse.p_lapse._param.requires_grad = False

    # p_cond__rt_ch = model.forward(ev)
    # loss = sim2d.fun_loss(p_cond__rt_ch, n_cond_rt_ch)
    # print(loss)

    if dict_subdir is None:
        dict_subdir = {
            'tddst': model.dtb.distrib_kind,
            'dispub': disper_ub,
            'tnd': model.tnds[0].kind,
            # 'tndloc': model.tnds[0].loc_kind,
            'tnddsp': model.tnds[0].disper_kind,
            'lps': model.lapse.kind,
            'lpsub': '%g' % npy(model.lapse.p_lapse.ub),
            'nfl': n_fold_test,
            'mdtrn': mode_train,
            'trst': trial_st,
            'thtr': model.dtb.thres_n_tr,
            'sm1': model.dtb.sumto1_wi_cond,
            # 'thn': thres_n_tr,
        }
        if correct_only:
            dict_subdir.update({'co': 1})
        if exclude_0coh:
            dict_subdir.update({'e0': 1})

    if dict_cache is None:
        dict_cache = {
            **dict_subdir,
            'sbj': subj,
            'prd': parad,
            'bim': int(bimanual),
            'ndim': 2,
            'td': model.dtb.kind,
            'ifl': i_fold_test,
        }

    if fit_mode == 'init_model':
        return model, data, dict_cache, dict_subdir

    cache = locfile1.get_cache('fit', dict_cache, subdir=dict_subdir)
    load_cache = cache.exists()

    kw = {
        'i_fold_test': i_fold_test,
        'n_fold_test': n_fold_test,
        'mode_train': mode_train,
        'to_debug': to_debug,
        'comment': argsutil.dict2fname(dict_cache),
        **kwargs
    }

    if load_cache:
        best_loss, best_state = cache.getdict([
            'best_loss', 'best_state'])
        d = cache.get()
        model.load_state_dict(best_state, strict=False)

        if fit_mode == 'd_only':
            return model, data, dict_cache, dict_subdir, d
        else:
            _, best_state, d, plotfuns = sim2d.fit_dtb(
                model, data,
                **{
                    **kw,
                    'max_epoch': 0,
                    'learning_rate': 0,
                }
            )
    else:
        best_loss, best_state, d, plotfuns = sim2d.fit_dtb(
            model, data, **kw
        )
        d1 = {k: v for k, v in d.items() if k.startswith('loss_')}
        d1.update({
            'best_loss': best_loss,
            'best_state': best_state
        })
        cache.set(d1)
        cache.save()
    del cache
    d['best_loss'] = best_loss
    d['best_state'] = best_state

    if to_save_res is None:
        to_save_res = not load_cache

    if to_save_res:
        def fun_tab_file(kind, ext):
            return locfile1.get_file('tab', kind, dict_cache, ext=ext,
                                    subdir=dict_subdir)

        def fun_fig_file(kind, ext):
            return locfile1.get_file_fig(kind, dict_cache, ext=ext,
                                        subdir=dict_subdir)

        ykt.save_optim_results(
            model, best_state, d,
            plotfuns=plotfuns, fun_tab_file=fun_tab_file,
            fun_fig_file=fun_fig_file)

        fit2d.plot_fit_combined(
            data, npy(d['out_all']), model,
            to_combine_ch_irr_cond=False,
            to_plot_internals=False,
            to_plot_params=False,
        )
        file = fun_fig_file('ch_rt', '.png')
        plt.savefig(file, dpi=72)
        print('Saved to %s' % file)

    print('--')
    return model, data, dict_cache, dict_subdir, d


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.double)

    main()