import numpy as np
import pandas as pd
try:
    from bottleneck import nanmean
except ImportError:
    from numpy import nanmean
from scipy.ndimage.filters import gaussian_filter1d

import mini_repo.classes.exceptions as exc

def get_heatmap(sig, running, abs_pos, n_bins, fp=1,
                normalized=False):
    n_laps = int(abs_pos[-1]) + 1

    heatmap = np.zeros((n_laps, n_bins))
    counts = np.zeros((n_laps, n_bins))

    for i in xrange(len(sig)):

        if not running[i]:
            continue
        if np.isnan(sig[i]):
            continue

        lap = int(abs_pos[i])
        posbin = int(np.mod(abs_pos[i] * n_bins, n_bins))

        heatmap[lap, posbin] += sig[i]
        counts[lap, posbin] += fp

    heatmap /= counts

    # Normalized
    if normalized and (np.nanmax(heatmap) > 0):
        heatmap = heatmap / np.nanmax(heatmap)

    return heatmap

def sensitivity(
        exptGrp, roi_filter=None, includeFrames='running_only'):
    """
    Fraction of complete forward passes through the place field that trigger
    a significant calcium transient

    returns a place_cell_df
    """

    pc_filter = exptGrp.pcs_filter(roi_filter=roi_filter)
    pfs_n = exptGrp.pfs(roi_filter=pc_filter)

    data_list = []
    for expt in exptGrp:
        tid = expt.trial_id
        mouse_name = expt.parent.mouse_name
        fov = expt.get('uniqueLocationKey')

        pfs = pfs_n[expt]
        if includeFrames == 'running_only':
            imaging_label = exptGrp.args['imaging_label']
            if imaging_label is None:
                imaging_label = expt.most_recent_key(
                    channel=exptGrp.args['channel'])
        for trial_idx, trial in enumerate(expt.findall('trial')):
            position = ba.absolutePosition(trial, imageSync=True)
            spikes = expt.spikes(roi_filter=pc_filter, label=exptGrp.args['imaging_label'], binary=True)
            running_kwargs = {'min_duration': 1.0, 'min_mean_speed': 0,
                      'end_padding': 0, 'stationary_tolerance': 0.5,
                      'min_peak_speed': 5, 'direction': 'forward'}

            running_frames = expt.runningIntervals(returnBoolList=True, **running_kwargs)[0]
            rois = expt.rois(
                roi_filter=pc_filter, channel=exptGrp.args['channel'],
                label=exptGrp.args['imaging_label'])

            spikes = trans

            assert len(rois) == len(pfs)
            assert len(rois) == len(spikes)

            for roi_spikes, roi_pfs, roi in it.izip(spikes, pfs, rois):

                pf_idx = np.zeros((100,))
                for roi_pf in roi_pfs:

                    if roi_pf[0] < roi_pf[1]:
                        pf_idx[roi_pf[0]:roi_pf[1] + 1] = 1.
                    else:
                        pf_idx[roi_pf[0]:] = 1.
                        pf_idx[:roi_pf[1]] = 1.
                pf_idx = pf_idx.astype(bool)

                heatmap = get_heatmap(roi_spikes, running_frames, position, 100)

                passes = heatmap.shape[0]
                hits = 0
                for lap in heatmap:
                    if np.nansum(lap[pf_idx]) > 0:
                        hits += 1

                data_dict = {'expt_id': tid,
                             'mouse': mouse_name,
                             'FOV': fov,
                             'roi': roi.label,
                             'value': float(hits) / passes}
                data_list.append(data_dict)

    return pd.DataFrame(data_list, columns=['expt_id', 'mouse', 'FOV', 'roi', 'value'])

def specificity(
        exptGrp, roi_filter=None, includeFrames='running_only', ignore_start=False):
    """
    Fraction of transient onsets that occur in a place field
    """
    pc_filter = exptGrp.pcs_filter(roi_filter=roi_filter)
    pfs_n = exptGrp.pfs(roi_filter=pc_filter)

    data_list = []
    for expt in exptGrp:
        tid = expt.trial_id
        mouse_name = expt.parent.mouse_name
        fov = expt.get('uniqueLocationKey')

        pfs = pfs_n[expt]
        if includeFrames == 'running_only':
            imaging_label = exptGrp.args['imaging_label']
            if imaging_label is None:
                imaging_label = expt.most_recent_key(
                    channel=exptGrp.args['channel'])
        for trial_idx, trial in enumerate(expt.findall('trial')):
            abspos = ba.absolutePosition(trial, imageSync=True)
            position = (abspos % 1) * 100
            spikes = expt.spikes(roi_filter=pc_filter, label=exptGrp.args['imaging_label'], binary=True)
            rois = expt.rois(
                roi_filter=pc_filter, channel=exptGrp.args['channel'],
                label=exptGrp.args['imaging_label'])

            running_kwargs = {'min_duration': 1.0, 'min_mean_speed': 0,
                      'end_padding': 0, 'stationary_tolerance': 0.5,
                      'min_peak_speed': 5, 'direction': 'forward'}

            running_frames = expt.runningIntervals(returnBoolList=True, **running_kwargs)[0]

            laps = abspos.astype(int)

            if ignore_start:
                interval = np.where(running_frames & (laps >= 6))[0]
            else:
                interval = np.where(running_frames)[0]

            

            assert len(rois) == len(pfs)
            assert len(rois) == len(spikes)

            position = position[interval]
            spikes = spikes[:, interval]

            for roi_spikes, roi_pfs, roi in it.izip(spikes, pfs, rois):

                pf_bins = []
                for roi_pf in roi_pfs:

                    pf_idx = np.zeros(roi_spikes.shape)

                    if roi_pf[0] < roi_pf[1]:
                        pf_idx = (position > roi_pf[0]) & (position <= roi_pf[1])
                    else:
                        pf_idx = (position > roi_pf[0]) | (position <= roi_pf[1])

                    pf_bins.append(pf_idx)
                pf_idx = reduce(np.logical_or, pf_bins)

                n_pf_spikes = np.nansum(roi_spikes[pf_idx])
                n_spikes = np.nansum(roi_spikes)

                data_dict = {'expt_id': tid,
                             'mouse': mouse_name,
                             'FOV': fov,
                             'roi': roi.label,
                             'value': n_pf_spikes / n_spikes}
                data_list.append(data_dict)
    return pd.DataFrame(data_list, columns=['expt_id', 'mouse', 'FOV', 'roi', 'value'])


def within_session_correlation(exptGrp, roi_filter=None, n_splits=100):

    pc_filter = exptGrp.pcs_filter(roi_filter=roi_filter)

    data_list = []
    for expt in exptGrp:
        tid = expt.trial_id
        mouse_name = expt.parent.mouse_name
        fov = expt.get('uniqueLocationKey')

        position = ba.absolutePosition(expt.find('trial'), imageSync=True)
        spikes = expt.spikes(roi_filter=pc_filter, label=exptGrp.args['imaging_label'], binary=True)
        rois = expt.rois(roi_filter=pc_filter, label=exptGrp.args['imaging_label'])

        running_kwargs = {'min_duration': 1.0, 'min_mean_speed': 0,
                      'end_padding': 0, 'stationary_tolerance': 0.5,
                      'min_peak_speed': 5, 'direction': 'forward'}

        running_frames = expt.runningIntervals(returnBoolList=True, **running_kwargs)[0]

        for roi, roi_spikes in zip(rois, spikes):

            heatmap = get_heatmap(roi_spikes, running_frames, position, 100)
            n_laps = heatmap.shape[0]
            half_laps = int(n_laps / 2)

            # Generate random splits
            corrs = []
            if n_splits == 1:
                tc1 = np.nan_to_num(np.nanmean(heatmap[::2, :], axis=0))
                tc2 = np.nan_to_num(np.nanmean(heatmap[1::2, :], axis=0))

                tc1 = gaussian_filter1d(tc1, 3, mode='wrap')
                tc2 = gaussian_filter1d(tc2, 3, mode='wrap')

                corrs.append(pearsonr(tc1, tc2)[0])

            else:
                for i in xrange(n_splits):
                    permutation = np.random.permutation(n_laps)

                    tc1 = np.nan_to_num(np.nanmean(heatmap[permutation[:half_laps], :], axis=0))
                    tc2 = np.nan_to_num(np.nanmean(heatmap[permutation[half_laps:], :], axis=0))

                    tc1 = gaussian_filter1d(tc1, 3, mode='wrap')
                    tc2 = gaussian_filter1d(tc2, 3, mode='wrap')

                    corrs.append(pearsonr(tc1, tc2)[0])

            data_dict = {'expt_id': tid,
                         'mouse': mouse_name,
                         'FOV': fov,
                         'roi': roi.label,
                         'value': np.nanmean(corrs),
                         'median_value': np.nanmedian(corrs)}
            data_list.append(data_dict)

    return pd.DataFrame(data_list, columns=['expt_id', 'mouse', 'FOV', 'roi', 'value', 'median_value'])


def place_cell_percentage(exptGrp, roi_filter=None, circ_var=False):
    """Calculate the percentage of cells that are a place cell on each day."""
    pcs_filter = exptGrp.pcs_filter(roi_filter=roi_filter, circ_var=circ_var)
    data_list = []
    for expt in exptGrp:
        n_rois = len(expt.rois(
            roi_filter=roi_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label']))
        n_pcs = len(expt.rois(
            roi_filter=pcs_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label']))

        if n_rois == 0:
            result = np.nan
        else:
            result = float(n_pcs) / n_rois

        data_dict = {'expt': expt,
                     'value': result}
        data_list.append(data_dict)
    return pd.DataFrame(data_list, columns=['expt', 'value'])


def place_field_width(exptGrp, roi_filter=None, belt_length=200):
    """Calculate all place field widths.

    Keyword arguments:
    exptGrp -- pcExperimentGroup to analyze
    belt_length -- length of the belt in cm

    Output: Pandas DataFrame consisting of one value per observation of a
        place field

    """

    pfs_n = exptGrp.pfs_n(roi_filter=roi_filter)
    data_list = []
    for expt in exptGrp:
        try:
            belt_length = expt.track_length / 10.
        except exc.NoBeltInfo:
            print 'No belt information found for experiment {}.'.format(
                str(expt))
            print 'Using default belt length = {}'.format(str(belt_length))

        rois = expt.rois(
            roi_filter=roi_filter, channel=exptGrp.args['channel'],
            label=exptGrp.args['imaging_label'])

        assert len(rois) == len(pfs_n[expt])

        for roi, pfs in zip(rois, pfs_n[expt]):
            if not len(pfs):
                continue
            for pf_idx, pf in enumerate(pfs):
                if pf[0] <= pf[1]:
                    value = (pf[1] - pf[0]) * belt_length
                else:
                    value = (1 + pf[1] - pf[0]) * belt_length

                data_dict = {'expt': expt,
                             'roi': roi,
                             'value': value}
                data_list.append(data_dict)

    return pd.DataFrame(data_list, columns=['expt', 'roi', 'value'])