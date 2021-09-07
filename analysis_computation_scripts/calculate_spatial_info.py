import numpy as np
from scipy.stats import percentileofscore
from scipy.signal import convolve2d

from collections import Counter
import itertools as it
from multiprocessing import Pool

import behavior_analysis as ba

import time

import pandas as pd

from dbclasses import dbExperiment

import cPickle as pkl

import sys


def _calc_information(
        MAX_N_POSITION_BINS, true_values, true_counts, bootstrap_values,
        bootstrap_counts, n_bins_list, smooth_lengths, n_processes):
    """
    Returns
    -------
    true_information : ndarray(ROIs, nbins)
    shuffle_information : ndarray(ROIs, bootstraps, nbins)
    """
    nROIs = len(true_counts)
    n_bootstraps = bootstrap_values.shape[2]
    true_information = np.empty((nROIs, len(n_bins_list)))
    shuffle_information = np.empty((nROIs, n_bootstraps, len(n_bins_list)))
    if n_processes > 1:
        pool = Pool(processes=n_processes)

    for bin_idx, (n_bins, factor_smoothing) in enumerate(zip(
            n_bins_list, smooth_lengths)):
        true_information_by_shift = np.empty(
            (nROIs, MAX_N_POSITION_BINS / n_bins))
        for bin_shift in np.arange(MAX_N_POSITION_BINS / n_bins):
            values = np.nansum(np.roll(true_values, shift=bin_shift, axis=1).reshape(
                [nROIs, n_bins, -1]), axis=2)
            counts = np.nansum(np.roll(true_counts, shift=bin_shift, axis=1).reshape(
                [nROIs, n_bins, -1]), axis=2)

            true_information_by_shift[:, bin_shift] = calc_spatial_information(
                (values, counts, factor_smoothing))

        true_information[:, bin_idx] = np.max(
            true_information_by_shift, axis=1)

        shuffle_information_by_shift = np.empty(
            (nROIs, n_bootstraps, MAX_N_POSITION_BINS / n_bins))
        for bin_shift in np.arange(MAX_N_POSITION_BINS / n_bins):

            shuffle_values = np.rollaxis(np.nansum(np.roll(
                bootstrap_values, shift=bin_shift, axis=1).reshape(
                [nROIs, n_bins, -1, n_bootstraps]), axis=2), 2, 0)

            assert np.all(np.around(
                np.std(np.sum(shuffle_values, axis=2), axis=0), 12) == 0)

            shuffle_counts = np.rollaxis(np.nansum(np.roll(
                bootstrap_counts, shift=bin_shift, axis=1).reshape(
                [nROIs, n_bins, -1, n_bootstraps]), axis=2), 2, 0)

            if n_processes > 1:
                chunksize = 1 + n_bootstraps / n_processes
                map_generator = pool.imap_unordered(
                    calc_spatial_information, zip(
                        shuffle_values, shuffle_counts,
                        it.repeat(factor_smoothing)),
                    chunksize=chunksize)
            else:
                map_generator = map(
                    calc_spatial_information, zip(
                        shuffle_values, shuffle_counts,
                        it.repeat(factor_smoothing)))

            idx = 0
            for info in map_generator:
                shuffle_information_by_shift[:, idx, bin_shift] = info
                idx += 1
            shuffle_information[:, :, bin_idx] = np.max(
                shuffle_information_by_shift, axis=2)

    if n_processes > 1:
        pool.close()
        pool.join()

    return true_information, np.rollaxis(shuffle_information, 1, 0)


def calc_spatial_information(inputs):
    event_counts, obs_counts, smooth_length = inputs
    # This is devived from Skaggs with some algebraic simplifications

    info = []
    for roi_events, roi_obs in it.izip(event_counts, obs_counts):

        O_sum = roi_obs.sum()
        idx = np.nonzero(roi_events)[0]

        roi_events = roi_events[idx]
        roi_obs = roi_obs[idx]

        E_sum = roi_events.sum()

        R = roi_events / roi_obs

        i = np.dot(roi_events, np.log2(R)) / E_sum - np.log2(E_sum / O_sum)

        info.append(i)

    return info


def circular_shuffle(spikes, position, bins):
    shuffle = np.empty(spikes.shape)

    # Remove spike time-points in bad position bins
    # So that we only shuffle across valid position bins
    good_idx = np.where(map(lambda x: x in bins, position))[0]
    bad_idx = [i for i in xrange(len(position)) if i not in good_idx]

    good_spikes = spikes[:, good_idx]
    good_shuffle = np.empty(good_spikes.shape)

    pivot = np.random.randint(good_spikes.shape[1])

    good_shuffle = np.hstack([good_spikes[:, pivot:],
                              good_spikes[:, :pivot]])

    shuffle[:, good_idx] = good_shuffle
    shuffle[:, bad_idx] = np.nan  # This isn't strictly necessary...

    return shuffle


def circular_shuffle_position(position, bins):
    shuffle = np.empty(position.shape)

    good_idx = np.where(map(lambda x: x in bins, position))[0]
    bad_idx = [i for i in xrange(len(position)) if i not in good_idx]

    good_pos = position[good_idx]

    pivot = np.random.randint(position.shape[0])

    good_shuffle = np.hstack([good_pos[pivot:],
                              good_pos[:pivot]])

    shuffle[good_idx] = good_shuffle
    shuffle[bad_idx] = position[bad_idx]

    return shuffle


def _shuffler(inputs):
    (shuffle_method, spikes, position, init_counts,
     frames_to_include, n_position_bins, bins) = inputs

    # TODO add option to shuffle by lap?
    if shuffle_method == 'circular':
        shuffle = np.full(spikes.shape, np.nan)
        # Only pass in running related intervals of spike signal
        shuffle[:, frames_to_include] = \
            circular_shuffle(spikes[:, frames_to_include], position, bins)

        shuffle_values, shuffle_counts = find_truth(shuffle, position, init_counts,
                                                    frames_to_include,
                                                    n_position_bins, bins,
                                                    return_square=False)

    # Though these should basically be equivalent...
    if shuffle_method == 'position_circular':
        shuffled_position = circular_shuffle_position(position, bins)
        shuffle_values, shuffle_counts = find_truth(spikes, shuffled_position, init_counts,
                                                    frames_to_include, n_position_bins, bins,
                                                    return_square=False)

    return shuffle_values, shuffle_counts


def _shuffle_bin_values(spikes, position, init_counts, frames_to_include,
                        n_processes, n_bootstraps, n_position_bins, bins,
                        shuffle_method='circular'):
    nROIs = spikes.shape[0]

    inputs = (shuffle_method, spikes, position, init_counts,
              frames_to_include, n_position_bins, bins)

    if n_processes > 1:
        pool = Pool(processes=n_processes)
        chunksize = 1 + n_bootstraps / n_processes
        map_generator = pool.imap_unordered(
            _shuffler, it.repeat(inputs, n_bootstraps), chunksize=chunksize)
    else:
        map_generator = map(_shuffler, it.repeat(inputs, n_bootstraps))

    bootstrap_values = np.empty((nROIs, n_position_bins, n_bootstraps))
    bootstrap_counts = np.empty((nROIs, n_position_bins, n_bootstraps))

    bootstrap_idx = 0

    for values, counts in map_generator:
        bootstrap_values[:, :, bootstrap_idx] = values
        bootstrap_counts[:, :, bootstrap_idx] = counts
        bootstrap_idx += 1

    if n_processes > 1:
        pool.close()
        pool.join()

    return bootstrap_values, bootstrap_counts


def generate_tuning_curve(spikes, n_position_bins, position,
                          bins=None, return_square=False):
    from bottleneck import nansum

    def _helper(_bin, positions, _spikes, square=False):
        idx = np.where(positions == _bin)[0]
        spike = _spikes[idx]
        if square:
            spike = np.square(spike)
        return nansum(spike)

    if bins is None:
        bad_bins = []
    else:
        bad_bins = [x for x in xrange(n_position_bins) if x not in bins]

    values = [_helper(_bin, position, spikes) for _bin in xrange(n_position_bins)]
    values = np.asarray(values)
    values[bad_bins] = np.nan
    if return_square:
        values_squared = [_helper(_bin, position, spikes, square=True) for _bin in xrange(n_position_bins)]
        values_squared = np.asarray(values_squared)
        values_squared[bad_bins] = np.nan
        return values, values_squared

    return values

def find_truth(spikes, position, init_counts,
               frames_to_include, n_position_bins, bins=None,
               return_square=False):
    """
    Returns
    -------
    true_values :
        spikes per position bin. Dims=(nROis, nBins)
    true_counts :
        Number of observation per position bin

    """

    nROIs = spikes.shape[0]

    # TODO Shouldn't these be indexed by cycle?
    true_values = np.zeros((nROIs, n_position_bins))
    true_values_squared = np.zeros((nROIs, n_position_bins))
    true_counts = np.zeros((nROIs, n_position_bins))

    for roi_idx, roi_spikes in it.izip(it.count(), spikes):

        v = generate_tuning_curve(
            spikes=spikes[roi_idx, frames_to_include],
            position=position,
            bins=bins,
            n_position_bins=n_position_bins,
            return_square=return_square)

        if return_square:
            v, v2 = v
            true_values_squared[roi_idx] += v2

        true_values[roi_idx] += v

        true_counts[roi_idx] = init_counts

        # Don't count position observations where signal was NaN
        nan_idx = np.where(np.isnan(spikes[roi_idx, frames_to_include]))[0]
        try:
            true_counts[roi_idx, position[nan_idx]] -= 1
        except IndexError:
            pass

    if return_square:
        return true_values, true_values_squared, true_counts

    return true_values, true_counts


def binned_positions(expt, frames_to_include, n_position_bins,
                     lap_threshold=0.2):
    """Calculate the binned positions for each cycle

    Returns
    -------
    position:
        position at imaging sampling rate
    counts:
        Counter object giving total occupancy at each bin

    """

    absolute_position = ba.absolutePosition(expt.find('trial'), imageSync=True)
    absolute_position = absolute_position[frames_to_include]

    laps = absolute_position.astype(int)
    n_laps = float(laps[-1] + 1)
    position = ((absolute_position % 1) * n_position_bins).astype(int)

    # Exclude position bins that appear in fewer than lap_threshold laps
    laps_per_bin = []
    for bin in xrange(n_position_bins):
        # Frames in which this bin was occupied
        idx = np.where(position == bin)[0]
        # Set of laps on which this bin was occupied
        bin_laps = set(laps[idx])
        # Total fraction of laps on which this bin was observed
        laps_per_bin.append(len(bin_laps) / n_laps)

    good_bins = [i for i, bin in enumerate(laps_per_bin)
                 if bin >= lap_threshold]

    counts = Counter(position)

    counts = np.array([counts[x] for x in xrange(n_position_bins)])

    return position, counts, good_bins




def spatial_info(expt, intervals='running', n_position_bins=100,
                    channel='Ch2', label=None,
                    smooth_length=3, n_bootstraps=100,
                    confidence=95, n_processes=1,
                    verbose=False):
    last_time = time.time()

    running_kwargs = {'min_duration': 1.0, 'min_mean_speed': 0,
                      'end_padding': 0, 'stationary_tolerance': 0.5,
                      'min_peak_speed': 5, 'direction': 'forward'}

    # Smear and re-binarize Spikes
    spikes = expt.spikes(label=label, channel=channel)
    spikes[spikes > 0] = 1
    nan_idx = np.isnan(spikes)

    # Convolve with boxcar
    w = np.ones((1, 3)).astype(float)
    spikes = convolve2d(np.nan_to_num(spikes), w / w.sum(), mode='same')

    # Re-binarize, and reset nans
    spikes[spikes > 0] = 1
    spikes[nan_idx] = np.nan

    nROIs, nFrames = spikes.shape


    # Choose intervals to include
    if intervals == 'all':
        frames_to_include = np.arange(nFrames)
    elif intervals == 'running':
        # frames_to_include = np.where(expt.velocity()[0] > 1)[0]
        frames_to_include = expt.runningIntervals(returnBoolList=True, **running_kwargs)[0]
    else:
        # Assume frames_to_include was passed in directly
        frames_to_include = intervals

    if verbose:
        time_now = time.time()
        print 'Time Taken: {}'.format(time_now - last_time)
        last_time = time_now
        print 'Binning Positions...'

    position, init_counts, bins = binned_positions(expt, frames_to_include,
                                                   n_position_bins)

    if verbose:
        print 'Time Taken: {}'.format(time.time() - last_time)
        print 'Finding Truth...'
    # Generate true tuning curves for data (Already calced!)
    with open(expt.placeFieldsFilePath(signal='spikes'), 'rb') as fp:
        pcdict = pkl.load(fp)[label]['undemixed']

    pc_idx = np.where(pcdict['pfs'])[0]


    true_values = pcdict['true_values'][pc_idx, :]
    true_counts = pcdict['true_counts'][pc_idx, :]

    # Generate bootstraps by shuffling spikes and computing tuning curves
    spikes = spikes[pc_idx, :]
    nROIs = len(pc_idx)

    bootstrap_values, bootstrap_counts = _shuffle_bin_values(
        spikes, position, init_counts, frames_to_include,
        n_processes=n_processes, n_bootstraps=n_bootstraps,
        n_position_bins=n_position_bins, bins=bins,
        shuffle_method='position_circular')

    n_bins_list = [2, 4, 5, 10, 20, 25, 50, 100]
    smooth_lengths = [3, 1, 1, 0, 0, 0, 0, 0]
    true_info, shuffle_info = _calc_information(n_position_bins, true_values, true_counts,
                                                    bootstrap_values, bootstrap_counts,
                                                    n_bins_list, smooth_lengths, n_processes=n_processes)

    shuffle_means = shuffle_info.mean(axis=0)  # rois x bins
    true_diffed = true_info - shuffle_means  # rois x bins
    shuffle_diffed = shuffle_info - shuffle_means  # bootstraps x rois x bins

    # take best bin for true and for each shuffle
    optimal_true_info = true_diffed.max(axis=1)
    optimal_shuffle_info = shuffle_diffed.max(axis=2)

    spatial_info_pct = [percentileofscore(optimal_shuffle_info[i, :], optimal_true_info[i]) for i in xrange(nROIs)]
    spatial_info_z = [(optimal_true_info[i] - np.nanmean(optimal_shuffle_info[i, :])) / np.nanstd(optimal_shuffle_info[i, :]) for i in xrange(nROIs)]

    data_list = {'expt_id': expt.trial_id,
                 'mouse': expt.parent.mouse_name,
                 'FOV': expt.get('uniqueLocationKey'),
                 'spatial_info': optimal_true_info,
                 'spatial_info_pct': np.array(spatial_info_pct),
                 'spatial_info_z': np.array(spatial_info_z)}

    return pd.DataFrame(data_list)


if __name__ == '__main__':

    save_path = './spatial_info/{}_{}.pkl'

    expt_dict = {'hb169_2': [20295, 20296, 20293, 20297],
                 'hb177_2': [20522, 20523, 20666, 20668, 20675],
                 'hb184_3': [20812, 20811, 20891, 20892, 20894, 20893, 20900, 20899],
                 'hb189_1': [21297, 21298, 21294, 21296, 21295, 21299, 21312, 21311],
                 'hb190_1': [21305, 21302, 21303, 21300, 21301, 21304, 21314, 21313],
                 'hb192_1': [21387, 21386, 21388, 21389, 21459, 21457, 21456, 21455]}

    mouse_name = sys.argv[1]

    for eid in expt_dict[mouse_name]:

        print eid

        expt = dbExperiment(eid)

        for label in ['drawn_red', 'drawn_green']:

            print label

            out = spatial_info(expt, label=label, smooth_length=3,
                               n_bootstraps=10000, n_processes=4)

            out.to_pickle(save_path.format(expt.trial_id, label))

