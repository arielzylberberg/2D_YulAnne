#  Copyright (c) 2020. Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
from matplotlib import pyplot as plt
from pprint import pprint
import numpy_groupies as npg

import torch

from lib.pylabyk import localfile, argsutil, np2

from data_2d import consts, load_data
from a0_dtb import a1_dtb_1D_sim as sim1d

# run "tensorboard --logdir runs" in the terminal to see log

locfile = localfile.LocalFile(
    pth_root='../../Data_2D/Data_2D_Py/a0_dtb'
)


def get_data_1D(i_subj=0, parad='RT', dim_rel=0, adcond_incl=None):
    # Choose by dim_rel and parad
    dim_irr = consts.get_odim(dim_rel)

    dat0 = load_data.load_data_combined()
    dat = np2.filt_dict(dat0, (
        (dat0['dim_rel'][:, dim_rel])
        & (~dat0['dim_rel'][:, dim_irr])
        & (dat0['id_parad'] == dat0['parads'].index(parad))
    ))

    # Choose subject
    id_subjs = np.unique(dat['id_subj'])
    print('Subjects included: ', end='')
    print(np.array(dat['subjs'])[id_subjs]) # to check
    id_subj = id_subjs[i_subj]
    subj = dat['subjs'][id_subj]
    dat = np2.filt_dict(dat, dat['id_subj'] == id_subj)

    # Choose dif_irr
    dif_irrs = np.unique(np.abs(dat['cond'][:, dim_irr]))
    if adcond_incl is None:
        adcond_incl = np.arange(len(dif_irrs))
    adcond_incl = list(adcond_incl)
    dif_irr = dif_irrs[adcond_incl]
    dat = np2.filt_dict(dat, np.any(
        np.abs(dat['cond'][:, [dim_irr]])
        == dif_irr[None, :]
        , axis=1
    ))

    # ev and ch
    incl = ~np.isnan(dat['ch'][:, dim_rel]) & ~np.isnan(dat['RT'])
    ch = (dat['ch'][incl, dim_rel] - 1).astype(np.long)
    rt = dat['RT'][incl]  # type: np.ndarray
    cond = dat['cond'][incl, dim_rel]

    return dat, ch, rt, cond, subj


def dat2p_dat(ch, rt, cond, nt=consts.NT, n_ch=consts.N_CH):
    drt = sim1d.rt_sec2fr(rt)

    conds, dcond = np.unique(cond, return_inverse=True)
    n_cond = len(conds)
    ev_cond = torch.tensor(conds)[:, None].expand([n_cond, nt])

    n_cond__rt_ch_dat = torch.tensor(npg.aggregate(
        np.stack([dcond, drt, ch.astype(np.long)]),
        1., 'sum', [n_cond, nt, consts.N_CH]
    ))
    return n_cond__rt_ch_dat, ev_cond, conds, dcond


def main_fit(
        model: sim1d.FitRT1D = sim1d.FitRT1D,
        i_subj=0, parad='RT', dim_rel=0, adcond_incl=None,
        n_fold_valid=5,
        subsample_factor=1,
        kw_name=(),
        ignore_cache=False,
        **kwargs
) -> (torch.nn.Module, torch.Tensor, dict, float, dict):
    """

    :param i_subj:
    :param parad:
    :param dim_rel:
    :param adcond_incl:
    :param n_fold_valid:
    :param bound_asymptote_max:
    :param ignore_cache:
    :return: model, ev_cond, p_cond__rt_ch_dat, dat, best_loss, dict_cache
    """
    dt = consts.DT * subsample_factor
    nt = int(consts.NT // subsample_factor)
    if type(model) is type:
        model = model(dt=dt, nt=nt, **kwargs)

    kw_name = dict(kw_name)

    dat, ch, rt, cond, subj = get_data_1D(i_subj, parad, dim_rel, adcond_incl)

    # Histogram with the number of bins reduced by subsample_factor
    n_cond__rt_ch_dat, ev_cond, conds, dcond = dat2p_dat(ch, rt, cond)
    n_cond__rt_ch_dat = sim1d.subsample_ev(
        n_cond__rt_ch_dat,
        subsample_factor=subsample_factor
    )[0] * subsample_factor
    n_cond = len(conds)

    # ev_cond subsampled by subsample_factor
    ev_cond0 = ev_cond
    # stack mean_ev and var_ev
    ev_cond = torch.stack(
        list(sim1d.subsample_ev(ev_cond0, subsample_factor)),
        -1
    )

    # Find cache and fit the model if there is none.
    tnd = model.tnds[0]
    dict_cache = argsutil.kwdef(
        kw_name, {
            'sbj': subj,
            'prd': parad,
            'dim': dim_rel,
            'odif': adcond_incl,
            'nfldv': n_fold_valid,
            'subsmp': subsample_factor,
            'tnd': tnd.kind,
            'tndloc': tnd.loc_kind,
            'tnddsp': tnd.disper_kind,
            'lps': model.lapse.kind
        }
    )
    cache = locfile.get_cache('fit', dict_cache)
    if cache.exists() and not ignore_cache:
        best_loss, best_state = cache.getdict(['best_loss', 'best_state'])
    else:
        print('Cache not found at %s\n= %s\nFitting new..'
              % (cache.fullpath, cache.fullpath_orig))
        # # Test run
        # p_rt_ch_pred = model(ev_cond)
        # cost = dtb.fun_loss(p_rt_ch_pred, p_rt_ch_dat)
        # cost.backward()
        # pprint([(v[0], v[1].data, v[1].grad) for v in
        #         model.named_parameters()])

        best_loss, best_state, d, plotfuns = sim1d.fit_dtb(
            model, ev_cond, n_cond_rt_ch_data=None,
            rt=np.clip((rt / dt).astype(np.long), 1, nt),
            ch=ch,
            cond=dcond,
            n_cond=n_cond,
            n_fold_valid=n_fold_valid,
            subsample_factor=subsample_factor
        )
        d = {k: v for k, v in d.items() if k.startswith('loss_')}
        d.update({
            'best_loss': best_loss,
            'best_state': best_state
        })
        cache.set(d)
        cache.save()

        print('model (fit):')
        print(model.__str__())

    p_cond__rt_ch_pred = model(ev_cond)
    cost = sim1d.fun_loss(p_cond__rt_ch_pred, n_cond__rt_ch_dat)
    cost.backward()
    pprint([(v[0], v[1].data, v[1].grad) for v in
            model.named_parameters()])

    model.load_state_dict(best_state)
    return model, ev_cond, n_cond__rt_ch_dat, dat, best_loss, dict_cache


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.set_num_threads(6)
    torch.set_default_dtype(torch.double)

    for i_subj in range(3):
        for dim_rel in range(2):
            if i_subj == 0 and dim_rel == 0:
                lapse_max=0.15
            else:
                lapse_max=1e-3

            model, ev_cond, p_rt_ch_dat, dat, best_loss, dict_cache = main_fit(
                i_subj=i_subj,
                dim_rel=dim_rel,
                ignore_cache=False,
                n_fold_valid=5,
                lapse_max=lapse_max,
                bound_asymptote_max=0.5,
                tnds=sim1d.TndInvNorm,
                lapse=sim1d.LapseUniform,
                subsample_factor=5,
                kw_name={
                },
            )
            # model = dtb.FitRT1D()  # CHECKED
            model.train()
            model.zero_grad()

            p_rt_ch_pred = model(ev_cond)
            loss = sim1d.fun_loss(p_rt_ch_pred, p_rt_ch_dat)
            loss.backward()

            fig = plt.figure(figsize=[6, 6])
            gs = plt.GridSpec(
                nrows=5, ncols=2,
                bottom=0.1, top=0.93,
                left=0.12, right=0.99,
                wspace=1.0, hspace=0.2,
                height_ratios=[1, 1, 0.3, 1, 1],
                width_ratios=[0.4, 0.6]
            )

            ax = plt.subplot(gs[0, 0])
            sim1d.plot_p_ch_vs_ev(ev_cond, p_rt_ch_pred, style='pred', color='k')
            sim1d.plot_p_ch_vs_ev(ev_cond, p_rt_ch_dat, style='data', mfc='k')
            ax.set_xlabel('')
            ax.set_xticklabels([])

            plt.subplot(gs[1, 0])
            sim1d.plot_rt_vs_ev(ev_cond, p_rt_ch_pred, style='pred')
            sim1d.plot_rt_vs_ev(ev_cond, p_rt_ch_dat, style='data')

            ax = plt.subplot(gs[3, 0])
            model.plot_p_tnd(ch=[0, 1])
            ax.set_xlabel('')
            ax.set_xticklabels([])

            ax = plt.subplot(gs[4, 0])
            model.plot_bound(color='k')

            ax = plt.subplot(gs[:, 1])
            model.plot_params()

            plt.suptitle('Subj %s, %s (%s, 1D)'
                         % (dict_cache['sbj'],
                            consts.DIM_NAMES_LONG[dict_cache['dim']],
                            dict_cache['prd']))

            file = locfile.get_file_fig('fit_combined', dict_cache)
            plt.savefig(file, dpi=150)
            print('Saved to %s' % file)
            plt.show()