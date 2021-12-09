import shelve
from collections import defaultdict
import os
import re
import time
import numpy as np
import pandas as pd
import time
import glob
import argparse


def is_match(x):
    return re.search(args.grep, os.path.basename(x)) is not None


parser = argparse.ArgumentParser()
parser.add_argument('exp_dir', type = str)
parser.add_argument('-cw', '--col-width', type=int, default=100)
parser.add_argument('-g', '--grep', type=str, default=None)
args = parser.parse_args()
print(args)

print('exp_dir=', args.exp_dir)

files = sorted(list(map(lambda x: x[:-4], glob.glob(os.path.join(args.exp_dir, '*.dat')))))

if args.grep is not None:
    files = list(filter(is_match, files))
    print(f'{len(files)} files matched')


results = defaultdict(list)
columns=[
        'epoch',
         *['R@{}'.format(i) for i in [1, 2, 4, 8, 16]],
         'NMI',
         'R@1-penult',
        ]

last_modified = None


is_valid = dict()

for p in files:
    is_valid[p] = False

    try:
        db = shelve.open(p)
        log_path = p + '.log'
        assert os.path.exists(log_path), log_path
        last_modified = (time.time() - os.path.getmtime(p + '.log')) / 60
        if not len(db):
            raise ValueError('Empty DB')
    except Exception as e:
        print('Failed to open', p, f'-- [{e}]')
        continue

    try:
        cur_results = np.vstack([
             np.array([epoch, *d['score']['eval']['final'][1], d['score']['eval']['final'][0]]) \
             for (epoch, d) in db['metrics'].items() if 'score' in d
        ])
    except Exception as e:
        print('Exception:', os.path.basename(p), e)
        continue
    try:
        cur_recall_penultimate = np.array([d['score']['eval']['penultimate'][1][0] \
                             for (epoch, d) in db['metrics'].items()])
    except:
        cur_recall_penultimate = np.array([np.nan] * len(cur_results))

    is_valid[p] = True

    idx_max_recall = cur_results[:, 1].argmax()
    best_epoch_results = cur_results[idx_max_recall]
    max_epoch = cur_results[:, 0].max()
    best_epoch_results = best_epoch_results.tolist()
    best_epoch_results[0] = '{:02}/{:02}'.format(int(best_epoch_results[0]), int(max_epoch))
    best_epoch_results.append(cur_recall_penultimate[idx_max_recall]) # R@1-penult
    if len(best_epoch_results) == len(columns) - 1:
        # R@16 is missing
        best_epoch_results = best_epoch_results[:5] + [np.nan] + best_epoch_results[5:]


    for i, col_name in enumerate(columns):
        results[col_name].append(best_epoch_results[i])

    dataset_name = os.path.basename(p).split('-', 1)[0]
    # if the file was last modified < 10 minute ago; than print Running status
    extra_timeout = {
        'vid': 40,
        'sop' : 15
    }
    if last_modified is None:
        results['S'].append('?')
    elif last_modified > (10 + extra_timeout.get(dataset_name, 0) * 15):
        results['S'].append('-')
    else:
        results['S'].append('[R]')


df = pd.DataFrame(index=list(map(os.path.basename, [f for f in files if is_valid[f]])),
                  data=results)

pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.max_colwidth', args.col_width)
pd.set_option('display.width', 1000000)
df.sort_values(by=['R@1'], inplace=True)
print(df)
