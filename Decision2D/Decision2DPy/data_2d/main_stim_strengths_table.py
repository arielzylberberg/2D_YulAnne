#  Copyright (c) 2020 Yul HR Kang. hk2699 at caa dot columbia dot edu.


from typing import Dict, Union, Any, Tuple, Iterable, List
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from pprint import pprint
from collections import OrderedDict as odict
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter)
import torch
import csv

from lib.pylabyk import localfile, np2, plt2, argsutil, yktorch as ykt
from lib.pylabyk import numpytorch as npt
from lib.pylabyk.numpytorch import npy
from lib.pylabyk.cacheutil import mkdir4file

npt.device0 = torch.device('cpu')
ykt.default_device = torch.device('cpu')

from data_2d import load_data, consts

locfile = localfile.LocalFile(
    pth_root='../../Data_2D/Data_2D_Py/data_2d/main_stim_strengths_table',
    cache_dir=''
)


def main():
    save_conds_lumped()
    save_conds_per_dim()


def save_conds_lumped():
    dat = load_data.load_data_combined()

    file = locfile.get_file_csv('conds')
    mkdir4file(file)
    wrote_header = False

    with open(file, 'w') as csvfile:
        for parad in ['RT', 'sh', 'VD', 'unimanual', 'bimanual']:
            for subj in consts.SUBJS[parad]:
                parad1, bi = consts.parad2parad_bi(parad)
                i_subj = dat['subjs'].index(subj)
                i_parad = dat['parads'].index(parad1)
                dat1 = np2.filt_dict(
                    dat,
                    np.array(dat['id_subj'] == i_subj) &
                    np.array(dat['id_parad'] == i_parad)
                )
                conds = [np.unique(np.abs(dat1['cond'][:, i]))
                         for i in range(2)]
                conds_str = [';'.join(['%g' % c for c in conds1])
                             for conds1 in conds]
                n_runs = len(np.unique(dat1['i_all_Run']))

                d = {
                    'Paradigm': parad,
                    'Participant': subj,
                    'Motion strengths': conds_str[0],
                    'Color strengths': conds_str[1],
                    '# runs': len(np.unique(dat1['i_all_Run'])),
                    '# trials': len(dat1['i_all_Run']),
                }
                if not wrote_header:
                    writer = csv.DictWriter(csvfile, fieldnames=d.keys())
                    writer.writeheader()
                    wrote_header = True

                writer.writerow(d)

    print('Saved to %s' % file)
    print('--')


def save_conds_per_dim():
    dat = load_data.load_data_combined()

    file = locfile.get_file_csv('conds_by_dim')
    mkdir4file(file)
    wrote_header = False

    with open(file, 'w') as csvfile:
        for parad in ['RT', 'sh', 'VD', 'unimanual', 'bimanual']:
            for subj in consts.SUBJS[parad]:
                for n_dim_rel in [1, 2]:
                    for dim_rel in range(2):

                        parad1, bi = consts.parad2parad_bi(parad)
                        i_subj = dat['subjs'].index(subj)
                        i_parad = dat['parads'].index(parad1)
                        dat1 = np2.filt_dict(
                            dat,
                            np.array(dat['id_subj'] == i_subj) &
                            np.array(dat['id_parad'] == i_parad) &
                            np.array(dat['n_dim_task'] == n_dim_rel) &
                            np.array(dat['dim_rel'][:, dim_rel])
                        )
                        conds = [np.unique(np.abs(dat1['cond'][:, i]))
                                 for i in range(2)]
                        conds_str = [';'.join(['%g' % c for c in conds1])
                                     for conds1 in conds]

                        d = {
                            'Paradigm': parad,
                            'Participant': subj,
                            '# dimensions': n_dim_rel,
                            'Relevant dim': dim_rel,
                            'Motion strengths': conds_str[0],
                            'Color strengths': conds_str[1],
                            '# runs': len(np.unique(dat1['i_all_Run'])),
                            '# trials': len(dat1['i_all_Run']),
                        }
                        if not wrote_header:
                            writer = csv.DictWriter(csvfile, fieldnames=d.keys())
                            writer.writeheader()
                            wrote_header = True

                        writer.writerow(d)

    print('Saved to %s' % file)
    print('--')


if __name__ == '__main__':
    main()