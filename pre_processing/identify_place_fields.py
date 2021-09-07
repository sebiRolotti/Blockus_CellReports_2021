import numpy as np
from scipy.stats import percentileofscore, mode
from scipy.signal import convolve2d

from collections import Counter
import itertools as it
from operator import itemgetter
from multiprocessing import Pool

import behavior_analysis as ba

import time

import warnings

import sys
import os


def smooth_tuning_curves(tuning_curves, smooth_length=3, nan_norm=True, axis=-1):

    # mean_zeroed = np.array(tuning_curves)
    # mean_zeroed[np.isnan(tuning_curves)] = 0
    mean_zeroed = np.nan_to_num(tuning_curves)
    gaus_mean = gaussian_filter1d(mean_zeroed, smooth_length, mode='wrap', axis=axis)

    if nan_norm:
        isfinite = np.isfinite(tuning_curves).astype(float)
        sm_isfinite = gaussian_filter1d(isfinite, smooth_length, mode='wrap', axis=axis)
        return gaus_mean / sm_isfinite
    else:
        return gaus_mean


def _consecutive_integers(a):
    # From python doc recipes

    out = []

    for k, g in it.groupby(enumerate(a), lambda (i, x): i - x):
        out.append(map(itemgetter(1), g))

    return out


def extend_pf(pf, tc, thresh=0.33):
    # Rescale TC so that min is 0, this way we ensure that
    # we always find an edge
    tc = (tc - np.min(tc)) / (np.max(tc) - np.min(tc))

    # Find first point out to the left below threshold
    try:
        left = np.max(np.where(tc[:pf[0]] <= thresh)[0])
    except ValueError:
        # Then the left edge must be around the circle
        left = pf[1] + np.max(np.where(tc[pf[1]:] <= thresh)[0])

    # Do the exact same thing for the right
    try:
        right = pf[1] + np.min(np.where(tc[pf[1]:] <= thresh)[0])
    except ValueError:
        right = np.min(np.where(tc[:pf[0]] <= thresh)[0])

    return [left, right]


def overlap(a, b, n_bins=100):
    if a[0] < a[1]:
        a_range = range(a[0], a[1] + 1 % n_bins)
    else:
        a_range = np.hstack([range(a[0], n_bins), range(a[1] + 1)])

    if b[0] < b[1]:
        b_range = range(b[0], b[1] + 1 % n_bins)
    else:
        b_range = np.hstack([range(b[0], n_bins), range(b[1] + 1)])

    inter = set(a_range).intersection(b_range)

    if inter:
        # If there is intersection, merge.
        # New edges are those not in the intersection.
        if a[0] in inter:
            lEdge = b[0]
        else:
            lEdge = a[0]

        if a[1] in inter:
            rEdge = b[1]
        else:
            rEdge = a[1]

        return [lEdge, rEdge]

    else:
        return []


def merge_pfs(pfs):
    if len(pfs) <= 1:
        return pfs

    # Check each pair of pfs for potential merge
    # If merge occurs, recursively restart process until no merges happen
    # Note we always merge into left position to maintain sort
    for i in xrange(len(pfs) - 1):
        for j in xrange(i + 1, len(pfs)):
            merged = overlap(pfs[i], pfs[j])
            if merged:
                new_pfs = list(pfs)
                new_pfs[i] = merged
                del new_pfs[j]
                return merge_pfs(new_pfs)
            else:
                continue

    return pfs


def extend_and_merge_pfs(pfs, tcs):
    new_pfs = []

    for cell_pfs, cell_tc in zip(pfs, tcs):

        new_cell_pfs = []
        for cell_pf in cell_pfs:
            new_cell_pfs.append(extend_pf(cell_pf, cell_tc))

        # Potentially merge new pfs
        new_pfs.append(merge_pfs(new_cell_pfs))

    return new_pfs


def find_pfs(true_tc, bootstrap_tc, confidence, bins,
             n_position_bins, min_run=5, max_nans=1):
    # max_nans tells us how many excluded bins we are willing to interpolate
    # place fields over. We can deal with this up front by saying a 'bad bin'
    # was significant only if it is in a string of bad_bins shorter than max_nans
    # and the included bins on each side of this string were significant

    significant = true_tc > np.nanpercentile(bootstrap_tc, confidence, axis=2)

    # significant = np.zeros((significant_good_bins.shape[0], n_position_bins))
    # significant[:, bins] = significant_good_bins

    bad_bins = [x for x in xrange(n_position_bins) if x not in bins]

    if bad_bins:
        bad_runs = _consecutive_integers(bad_bins)
        # Check for circular bad run
        if (bad_runs[-1][-1] == n_position_bins - 1) and (bad_runs[0][0] == 0):
            bad_runs[0] = bad_runs[-1] + bad_runs[0]
            del bad_runs[-1]

        for bad_run in bad_runs:
            if len(bad_runs) > max_nans:
                continue

            # left and right border indices
            # (account for potential circularity)
            LB = np.mod(bad_run[0] - 1, n_position_bins)
            RB = np.mod(bad_run[-1] + 1, n_position_bins)

            for roi in significant:
                if roi[LB] and roi[RB]:
                    roi[bad_run[0]:bad_run[-1] + 1] = 1

    pfs = []

    for roi, roi_tc in zip(significant, true_tc):

        active_bins = np.where(roi)[0]
        runs = _consecutive_integers(active_bins)

        try:
            # Check for circular consecutive bins
            if (runs[-1][-1] == n_position_bins - 1) and (runs[0][0] == 0):
                runs[0] = runs[-1] + runs[0]
                del runs[-1]
        except IndexError:
            pass

        # PFs are start/stop of consecutive runs
        roi_pfs = [[r[0], r[-1]] for r in runs if len(r) >= min_run]

        # Sort pfs by peak activity within pf
        peaks = [np.nanmax(roi_tc[np.array(r)]) for r in runs
                 if len(r) >= min_run]

        roi_pfs = [rpf for rpf, _ in sorted(zip(roi_pfs, peaks),
                                            key=lambda pair: pair[1], reverse=True)]

        pfs.append(roi_pfs)

    return pfs


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
    print (laps)
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


def validate_pfs(pfs, spikes, expt, frames_to_include, n_position_bins, min_frac=0.15):
    absolute_position = ba.absolutePosition(expt.find('trial'), imageSync=True)
    absolute_position = absolute_position[frames_to_include]

    laps = absolute_position.astype(int)
    unique_laps = np.unique(laps)
    n_laps = len(unique_laps)
    position = ((absolute_position % 1) * n_position_bins).astype(int)

    spikes = spikes[:, frames_to_include]

    good_pfs = []
    good_laps = []

    for roi, roi_pfs in zip(spikes, pfs):

        good_roi_pfs = []

        for roi_pf in roi_pfs:

            if roi_pf[0] < roi_pf[1]:
                pf_idx = np.arange(roi_pf[0], roi_pf[1] + 1)
            else:
                pf_idx = np.hstack([np.arange(0, roi_pf[1] + 1),
                                    np.arange(roi_pf[0], n_position_bins)])

            n_good_laps = 0.

            for lap in unique_laps:

                lap_idx = np.where(laps == lap)[0]

                tc = generate_tuning_curve(
                    spikes=roi[lap_idx],
                    position=position[lap_idx],
                    n_position_bins=n_position_bins)

                # Check to see if mean activity in pf is greater than
                # overall mean
                if np.nansum(tc[pf_idx]) > 0:
                    n_good_laps += 1
                    good_laps.append(lap)
                # if np.nanmean(tc[pf_idx]) > np.nanmean(tc):
                #     n_good_laps += 1

            if (n_good_laps / n_laps) > min_frac:
            # if n_good_laps >= 3:
                good_roi_pfs.append(roi_pf)

        good_pfs.append(good_roi_pfs)

    return good_pfs


def id_place_fields(expt, intervals='running', n_position_bins=100,
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
            running_intervals = expt.runningIntervals(returnBoolList=False,
                                                      **running_kwargs)

            if not len(running_intervals[0]):
                return []
            # list of frames, one per list element per cycle
            frames_to_include = np.hstack([np.arange(start, end) for
                                           start, end in running_intervals[0]])
        # frames_to_include = np.where(expt.velocity()[0] > 1)[0]
    else:
        # Assume frames_to_include was passed in directly
        frames_to_include = intervals

    # Only analyze post-stim intervals, for induction based experiments

    if len(frames_to_include) == 0:
        return None
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
    # Generate true tuning curves for data
    true_values, true_values_squared, true_counts = \
        find_truth(spikes, position, init_counts, frames_to_include,
                   n_position_bins, bins, return_square=True)

    if verbose:
        time_now = time.time()
        print 'Time Taken: {}'.format(time_now - last_time)
        last_time = time_now
        print 'Strapping Boots...'
    # Generate bootstraps by shuffling spikes and computing tuning curves
    bootstrap_values, bootstrap_counts = _shuffle_bin_values(
        spikes, position, init_counts, frames_to_include,
        n_processes=n_processes, n_bootstraps=n_bootstraps,
        n_position_bins=n_position_bins, bins=bins,
        shuffle_method='position_circular')

    true_tc = true_values / true_counts
    bootstrap_tc = bootstrap_values / bootstrap_counts

    true_tc_smooth = smooth_tuning_curves(true_tc,
                                              smooth_length=smooth_length,
                                              nan_norm=False, axis=1)
    bootstrap_tc_smooth = smooth_tuning_curves(bootstrap_tc,
                                                   smooth_length=smooth_length,
                                                   nan_norm=False, axis=1)

    if verbose:
        time_now = time.time()
        print 'Time Taken: {}'.format(time_now - last_time)
        last_time = time_now
        print 'Finding Place Fields...'

    pfs = find_pfs(true_tc_smooth, bootstrap_tc_smooth, confidence,
                   bins, n_position_bins=n_position_bins,
                   min_run=5, max_nans=1)

    # Sort PFs by peak amplitude

    # Post hoc check: Only include PFs with increased in-pf activity for
    # at least some fraction of laps

    if verbose:
        time_now = time.time()
        print 'Time Taken: {}'.format(time_now - last_time)
        last_time = time_now
        print 'Validating Place Fields...'

    pfs = extend_and_merge_pfs(pfs, true_tc_smooth)

    pfs = validate_pfs(pfs, spikes, expt, frames_to_include,
                       n_position_bins)

    if verbose:
        time_now = time.time()
        print 'Time Taken: {}'.format(time_now - last_time)
        last_time = time_now
        print 'Book-Keeping...'

    # Save everything into result dictionary
    params = {'intervals': intervals, 'n_position_bins': n_position_bins,
              'smooth_length': smooth_length, 'running_kwargs': running_kwargs,
              'n_bootstraps': n_bootstraps, 'confidence': confidence}

    result = {'spatial_tuning': true_tc,
              'true_values': true_values,
              'true_counts': true_counts,
              'std': np.sqrt(true_values_squared / true_counts -
                             (true_values / true_counts) ** 2),
              'pfs': pfs,
              'parameters': params}
    if intervals == 'running':
        result['running_kwargs'] = running_kwargs
    if smooth_length > 0:
        result['spatial_tuning_smooth'] = true_tc_smooth
        result['std_smooth'] = smooth_tuning_curves(
            result['std'], smooth_length=smooth_length, nan_norm=False)

    time_now = time.time()
    print 'Time Taken: {}'.format(time_now - last_time)

    return result
