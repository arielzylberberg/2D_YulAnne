#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.

import numpy as np
from typing import Any
import csv
import statsmodels.api as sm

import torch

from lib.pylabyk import numpytorch as npt, yktorch as ykt
from lib.pylabyk import localfile, np2
from lib.pylabyk.cacheutil import mkdir4file

from data_2d import load_data, consts

npt.device0 = torch.device('cpu')
ykt.default_device = torch.device('cpu')

locfile = localfile.LocalFile(
    pth_root='../../Data_2D/Data_2D_Py/a5_GLM/a1_main_ch_by_odif',
    cache_dir=''
)


def main(parads=('RT', 'unimanual')):

    file = locfile.get_file_csv('logistic_ch_by_odif')
    mkdir4file(file)
    with open(file, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            'parad', 'dim_rel', 'dev', 'dev0', 'ddev', 'bic', 'bic0', 'dbic'
        ])
        writer.writeheader()

        for parad in parads:
            dat = load_data.load_data_combined()

            if parad in ['unimanual', 'bimanual']:
                filt = (
                    (np.array(dat['id_parad'] == dat['parads'].index(
                        'unibimanual')))
                    & (dat['bimanual'] == (parad == 'bimanual'))
                )
            else:
                filt = dat['id_parad'] == dat['parads'].index(parad)

            dat1 = np2.filt_dict(
                dat,
                filt & (np.array(dat['n_dim_task']) == 2)
                & (np.array(dat['id_subj']) != dat['subjs'].index('FR'))
            )
            print('parad: ' + parad)

            devs = []
            dev0s = []

            bics = []
            bic0s = []

            parad_label = 'eye_RT' if parad == 'RT' else parad

            for dim_rel in range(2):
                print('dim_rel: %d' % dim_rel)

                glmres, glmres0 = fit_ch_ixn_odif(dat1, dim_rel)[:4]
                devs.append(glmres.deviance)
                dev0s.append(glmres0.deviance)
                bics.append(glmres.bic)
                bic0s.append(glmres0.bic)

                writer.writerow({
                    'parad': parad_label,
                    'dim_rel': dim_rel,
                    'dev': glmres.deviance,
                    'dev0': glmres0.deviance,
                    'ddev': glmres.deviance - glmres0.deviance,
                    'bic': glmres.bic,
                    'bic0': glmres0.bic,
                    'dbic': glmres.bic - glmres0.bic,
                })

            writer.writerow({
                'parad': parad_label,
                'dim_rel': 'sum01',
                'dev': np.sum(devs),
                'dev0': np.sum(dev0s),
                'ddev': np.sum(devs) - np.sum(dev0s),
                'bic': np.sum(bics),
                'bic0': np.sum(bic0s),
                'dbic': np.sum(bics) - np.sum(bic0s),
            })

            print('sum(devs) - sum(dev0s):')
            print(np.sum(devs) - np.sum(dev0s))

            print('sum(bics) - sum(bic0s):')
            print(np.array(bics) - np.array(bic0s))
            # print(bics)
            # print(bic0s)
            print(np.sum(bics) - np.sum(bic0s))

    print('Saved to %s' % file)
    print('--')


    # res = {
            #     'parad': parad,
            #     # 'dimRelevant': consts.DIM_NAMES_LONG[dim_rel],
            #     # 'deviance0': glmres0.deviance,
            #     # 'deviance1': glmres.deviance,
            #     # 'dDeviance': glmres.deviance - glmres0.deviance,
            #     # 'BIC0': glmres0.bic,
            #     # 'BIC1': glmres.bic,
            #     # 'dBIC': glmres.bic - glmres0.bic,
            #     'deviance0': glmres0.deviance,
            #     'deviance1': glmres.deviance,
            #     'dDeviance': glmres.deviance - glmres0.deviance,
            #     'BIC0': glmres0.bic,
            #     'BIC1': glmres.bic,
            #     'dBIC': glmres.bic - glmres0.bic,
            # }


def fit_ch_ixn_odif(
        dat: dict, dim_rel: int) -> (Any, Any):
    dim_irr = consts.get_odim(dim_rel)

    ch = dat['ch'][:, dim_rel] == 2
    regs = [
        dat['cond'][:, dim_rel],
        np.abs(dat['cond'][:, dim_irr]),
        dat['cond'][:, dim_rel] * np.abs(dat['cond'][:, dim_irr]),
    ]

    id_subjs = np.unique(dat['id_subj'])
    for id_subj in id_subjs[:-1]:
        regs.append(
            # dat['cond'][:, dim_rel] * np.abs(dat['cond'][:, dim_irr])
            np.array(dat['id_subj'] == id_subj, dtype=np.float)
        )

    reg = sm.add_constant(np.stack(regs, -1))
    glmmodel = sm.GLM(
        ch,
        reg, family=sm.families.Binomial()
    )
    glmres = glmmodel.fit()

    coef = glmres.params
    se_coef = glmres.bse
    dev = glmres.bic
    # dev = glmres.deviance

    glmmodel0 = sm.GLM(
        ch,
        np.concatenate([reg[:, :3], reg[:, 4:]], -1),
        family=sm.families.Binomial()
    )
    glmres0 = glmmodel0.fit()
    dev0 = glmres0.bic
    # dev0 = glmres0.deviance

    # print(coef)
    # print(se_coef)
    # print(coef / se_coef)
    # print(dev - dev0)

    # return coef, se_coef, dev, dev0, glmres, glmmodel
    return glmres, glmres0


if __name__ == '__main__':
    main()