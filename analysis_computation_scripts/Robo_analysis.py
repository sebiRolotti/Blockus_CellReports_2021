from mini_repo.classes.dbclasses import dbMouse
from mini_repo.classes.classes import pcExperimentGroup

import mini_repo.analysis.place_cell_analysis as pca
import mini_repo.analysis.imaging_analysis as ia

from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter, maximum_filter
from scipy.stats import mannwhitneyu, ttest_rel

import numpy as np

import pandas as pd

import os
import cPickle as pkl

# Utilities

def roi_label(x):

    try:
        rlabel = x['roi'].label
    except AttributeError:
        return x['roi']

    return rlabel

def fix_dfs(dfs):

    new_dfs = []

    if 'trial' in dfs[0].columns:
        for df in dfs:
            df['expt_id'] = df.apply(lambda x: x['trial'].parent.trial_id, axis=1)
            df['mouse'] = df.apply(lambda x: x['trial'].parent.parent.mouse_name, axis=1)
            df['FOV'] = df.apply(lambda x: x['trial'].parent.get('uniqueLocationKey'), axis=1)

            if 'roi' in df.columns:
                df['roi'] = df.apply(roi_label, axis=1)

            new_df = df.drop(columns=['trial'])

            new_dfs.append(new_df)

    elif 'expt' in dfs[0].columns:
        for df in dfs:
            df['expt_id'] = df.apply(lambda x: x['expt'].trial_id, axis=1)
            df['mouse'] = df.apply(lambda x: x['expt'].parent.mouse_name, axis=1)
            df['FOV'] = df.apply(lambda x: x['expt'].get('uniqueLocationKey'), axis=1)

            if 'roi' in df.columns:
                df['roi'] = df.apply(roi_label, axis=1)

            new_df = df.drop(columns=['expt'])
            new_dfs.append(new_df)

    else:
        for df in dfs:
            if 'roi' in df.columns:
                df['roi'] = df.apply(roi_label, axis=1)
                new_dfs.append(df)

    return new_dfs


def save_data(data, save_path):

    data = fix_dfs(data)

    with open(save_path, 'wb') as fp:
        pkl.dump(data, fp)


##############################################
# Init Experiment Groups
mice = ['hb169_2', 'hb177_2', 'hb184_3', 'hb189_1', 'hb190_1', 'hb192_1']
expts = []
for mouse in mice:
    expts.extend(dbMouse(mouse).imagingExperiments())

expts = [e for e in expts if 'GOL' not in e.get('tSeriesDirectory')]

# Filter!
# Filter /data2/Heike/hb177_2/190627/hb177_2_RF_FOV2_day2-001 due to data drop
bad_tseries = ['/data2/Heike/hb177_2/190627/hb177_2_RF_FOV2_day2-001']
expts = [e for e in expts if not np.any([x in e.get('tSeriesDirectory') for x in bad_tseries])]

print len(expts)

green_kwargs = {'imaging_label': 'drawn_green',
                'nPositionBins': 100,
                'channel': 'Ch2',
                'demixed': False,
                'pf_subset': None,
                'signal': 'spikes'}

red_kwargs = {'imaging_label': 'drawn_red',
                'nPositionBins': 100,
                'channel': 'Ch2',
                'demixed': False,
                'pf_subset': None,
                'signal': 'spikes'}

wt_grp = pcExperimentGroup(expts, **green_kwargs)
ko_grp = pcExperimentGroup(expts, **red_kwargs)

grps = [wt_grp, ko_grp]

# grps = [pcExperimentGroup(expts[:2], **red_kwargs)]

colors = ['black', 'red']
colors = [sns.xkcd_rgb[x] for x in colors]

labels = ['WT', 'KO']

# 24 hour timepoint comparisons only:
longest = pd.Timedelta(pd.tseries.offsets.Hour(36))
shortest = pd.Timedelta(pd.tseries.offsets.Hour(12))


save_dir = './data'
##############################################

metrics = ['run_freq', 'silent', 'sensitivity', 'specificity', 'percentage', 'width', 'autocorr']


## Raw Activity
# Running
if 'run_freq' in plots:

    dfs = []
    for grp in grps:
        dfs.append(ia.population_activity_new(grp, 'binary_spike_frequency', interval='running'))

    save_data(dfs, os.path.join(data_dir, 'run_freq.pkl'))

if 'silent' in plots:

    dfs = []
    for grp in grps:
        dfs.append(ia.silent_cell_percentage(grp, 'binary_spike_frequency',
                                             label=grp.args['imaging_label'], interval=None))

    save_data(dfs, os.path.join(data_dir, 'silent.pkl'))

## Sensitivity
if 'sensitivity' in plots:

    dfs = []
    for grp in grps:
        dfs.append(pca.sensitivity(grp))

    save_data(dfs, os.path.join(data_dir, 'sensitivity.pkl'))

## Specificity
if 'specificity' in plots:

    dfs = []
    for grp in grps:
        dfs.append(pca.specificity(grp))

    save_data(dfs, os.path.join(data_dir, 'specificity.pkl'))

# Percent PFs
# Note: Already Grouped
if 'percentage' in plots:

    dfs = []
    for grp in grps:
        dfs.append(pca.place_cell_percentage(grp))
    save_data(dfs, os.path.join(data_dir, 'percentage.pkl'))

# Place Field Width
if 'width' in plots:

    dfs = []
    for grp in grps:
        dfs.append(pca.place_field_width(grp))
    save_data(dfs, os.path.join(data_dir, 'width.pkl'))


# if 'info' in plots:

#     red_df = pd.DataFrame([])
#     green_df = pd.DataFrame([])

#     dirname = '/home/sebi/spatial_info'

#     for filename in os.listdir(dirname):

#         if 'red' in filename:
#             df = pd.read_pickle(os.path.join(dirname, filename))
#             red_df = pd.concat([red_df, df])
#         elif 'green' in filename:
#             df = pd.read_pickle(os.path.join(dirname, filename))
#             green_df = pd.concat([green_df, df])

#     def log_p(x):

#         pval = 1 - x['spatial_info_pct'] / 100.
#         if pval == 0:
#             pval = 1 / 10000.

#         return np.log(pval)

#     green_df['value'] = green_df.apply(log_p, axis=1)
#     red_df['value'] = green_df.apply(log_p, axis=1)

#     dfs = [green_df, red_df]
#     grouped_dfs = []
#     for df in dfs:
#         grouped_dfs.append(df.groupby(['expt_id', 'mouse', 'FOV'], as_index=False).mean())

#     save_path = os.path.join(save_dir, 'spatial_info_logp_{{plot}}{}'.format(ending))
#     plot_cdf(grouped_dfs, labels, colors, save_path=save_path)
#     comparison_plot(grouped_dfs, labels, colors, ylabel='Spatial Info (log p-value)', save_path=save_path)


if 'autocorr' in plots:

    dfs = []
    for grp in grps:
        dfs.append(pca.within_session_correlation(grp, n_splits=1))

    save_data(dfs, os.path.join(data_dir, 'autocorr.pkl'))

