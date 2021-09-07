import argparse
import sys
import os
import h5py

import cPickle as pkl
import itertools as it

import numpy as np
from pandas import Series, DataFrame
from scipy.signal import savgol_filter, detrend
from scipy.ndimage.filters import gaussian_filter1d

from mini_repo.classes.dbclasses import dbExperiment
from mini_repo.classes.exceptions import NoSimaPath

# Assumes that OASIS package has been installed and added to python path
# See installation instructions at https://github.com/j-friedrich/OASIS
import warnings

try:
    from oasis.functions import deconvolve
    from oasis.oasis_methods import oasisAR1
except ImportError:
    warnings.warn('OASIS must be locally installed')


def maxmin_filter(signal, window=300, sigma=5):
    """Calculate baseline as the rolling maximum of the rolling minimum of the
    smoothed trace

    Parameters
    ----------
    signal : array, size (n_ROIs, n_timepoints)
    window : int
        Optional, size of the rolling window for max/min/smoothing
    sigma : int
        Standard deviation of the gaussian smoothing kernel
    """

    kwargs = {'window': window, 'min_periods': int(window / 5),
              'center': True, 'axis': 1}

    smooth_signal = pd.DataFrame(signal).rolling(
        win_type='gaussian', **kwargs).mean(std=sigma)

    return smooth_signal.rolling(**kwargs).min().rolling(**kwargs).max().values


def already_processed(expt, channel, label):
    spikes_path = expt.spikesFilePath(channel=channel)

    if os.path.splitext(spikes_path)[1] == '.h5':
        if not os.path.exists(spikes_path):
            return False
        with h5py.File(spikes_path, 'r') as f:
            return label in f[channel].keys()

    else:
        try:
            with open(expt.spikesFilePath(channel=channel), 'rb') as file:
                spikes = pkl.load(file)
            return label in spikes.keys()

        except IOError:
            return False


def labelsToProcess(expt, channel, labels=None, raw=False):
    """List ROI labels in experiment, using keys in signals or dfof"""
    try:
        if raw:
            with open(expt.signalsFilePath(channel=channel), 'rb') as f:
                imData = pkl.load(f)
        else:
            with open(expt.dfofFilePath(channel=channel), 'rb') as f:
                imData = pkl.load(f)
        if labels is not None:
            return [label for label in labels if label in imData.keys()]
        else:
            return imData.keys()
    except (IOError, pkl.UnpicklingError):
        return []


def fill_nans(imData, max_interp=0):
    for roi_ix, roi in enumerate(imData):
        nans = np.where(np.isnan(roi))[0]

        if len(nans):
            interval_starts = np.hstack(
                [nans[0], nans[np.where(np.diff(nans) > 1)[0] + 1]])
            # exclusive of end for easy slicing later
            interval_ends = np.hstack(
                [nans[np.where(np.diff(nans) > 1)[0]], nans[-1]]) + 1

            interval_durations = interval_ends - interval_starts
            s = Series(imData[roi_ix])
            # for long intervals, draw from a gaussian
            for int_ix, start, end in it.izip(
                    it.count(), interval_starts, interval_ends):
                if interval_durations[int_ix] > \
                        max_interp:
                    imData[roi_ix][start:end] = \
                        np.random.normal(
                            s.mean(), s.std(),
                            interval_durations[int_ix])
                else:
                    pass
            # interpolate for the rest
            s = Series(imData[roi_ix])
            s.interpolate(inplace=True)

            if np.any(np.isnan(s)):
                for idx in np.where(np.isnan(s)):
                    s[idx] = np.random.normal(s.mean(),
                                              s.std())

            imData[roi_ix] = s.values

    return imData


def remove_outliers(x):
    for row in x:
        row[np.where(row < np.percentile(row, 1))] = np.nanmean(row)
    return x


def mad_threshold(signal, s=None, n_mad=3):
    if s is not None:
        signal_filtered = signal[s == 0]
    else:
        signal_filtered = signal
    thresh = np.nanmedian(signal_filtered) + np.nanmedian(np.abs(signal_filtered - np.nanmedian(signal_filtered))) * \
             n_mad

    return thresh


def z_threshold(signal, s=None, n_sigs=2):
    if s is not None:
        signal_filtered = signal[s == 0]
    else:
        signal_filtered = signal
    thresh = np.nanmean(signal_filtered) + n_sigs * np.nanstd(signal_filtered)

    return thresh


def filter_spikes(signal, s, n_mad=3):
    """Estimate the noise of the original trace using MADs and retain only \
    spikes whose energy exceeds an n_mad threshold above the trace median"""
    thresh = mad_threshold(signal, s, n_mad=n_mad)
    s[s < thresh] = 0
    return s


def reconvolve_spikes(s, g, w=200, tol=1e-09):
    """Given spikes and AR coefficient, generate denoised calcium trace"""

    # AR1 calcium kernel
    h = np.exp(np.log(g) * np.arange(w))

    # superimpose a scaled version of the kernel on the trace at each spike
    c = np.zeros_like(s)
    for t in np.where(s > tol)[0]:
        c[t:t + w] += s[t] * h[:min(w, len(s) - t)]

    return c


def oasis_spikes(expt, label=None, channel=None, g=None, raw=False,
                 savgol=False, mad_filter=None, baseline=[None, None],
                 refine=0, lam=7):

    if raw:
        signals = expt.imagingData(
            label=label, channel=channel)[..., 0].astype(float)
    else:
        signals = expt.imagingData(label=label, channel=channel,
                                   dFOverF='from_file')[..., 0].astype(float)

    spikes = {k: np.empty(signals.shape) for k in ['denoised_ca', 'spikes',
                                                   'baseline']}

    nan_idx = np.where(np.isnan(signals))
    signals = fill_nans(signals, int(1. / expt.frame_period()))
    signals = detrend(signals, axis=1, type='linear')

    if baseline[0] == -1.:
        print('Estimating minimax baseline...')
        window = int(baseline[1] / expt.frame_period())

        # TODO pass sigma as a parameter
        spikes['baseline'] = maxmin_filter(remove_outliers(signals),
                                           window=window, sigma=5)
        signals -= spikes['baseline']

    elif baseline[0] is not None:
        print('Estimating rolling baseline...')
        rolling_baseline = DataFrame(signals).rolling(
            int(baseline[1]), center=True, axis=1, min_periods=1).quantile(
            baseline[0]).values

        signals -= rolling_baseline

    signals = np.array(signals)

    if savgol:
        window = int(np.ceil(30. / expt.frame_period()))
        if np.mod(window, 2) == 0:
            window += 1
        filtered_baseline = savgol_filter(signals, window, 1)
        signals -= filtered_baseline

    for i, signal in enumerate(signals):

        try:
            spikes['denoised_ca'][i, :], s, _, g, _ = deconvolve(
                signal.squeeze(), g=[g], penalty=1)

            if mad_filter:
                spikes['denoised_ca'][i, :], s = oasisAR1(
                    signal.squeeze(), g=g, s_min=mad_threshold(signal.squeeze(), s, n_mad=mad_filter), lam=lam) #100

            spikes['spikes'][i, :] = s

        except ValueError:
            spikes['denoised_ca'][i, :] = np.nan
            spikes['spikes'][i, :] = np.nan
            spikes['baseline'][i, :] = np.nan
    p.end()
    spikes['signals'] = signals
    spikes['signals'][nan_idx] = np.nan
    spikes['signals'] = spikes['signals'][..., np.newaxis]
    spikes['denoised_ca'][nan_idx] = np.nan
    spikes['spikes'][nan_idx] = np.nan
    spikes['baseline'][nan_idx] = np.nan

    if savgol:
        spikes['baseline'] += filtered_baseline

    spikes['params'] = {'g': g,
                        'raw': raw,
                        'savgol': savgol,
                        'mad_filter': mad_filter,
                        'baseline': baseline,
                        'refine': refine,
                        'lam': lam}

    return spikes


def main(argv):
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "xml", action='store', type=str, default='behavior.xml',
        help="name of xml file to parse")
    argParser.add_argument(
        "-m", "--mouse", action='store', type=str, nargs='+',
        help="enter mouse names to process spikes for")
    argParser.add_argument(
        "-d", "--trial_id", action="store", type=int, default='',
        help="Process experiment with given trial id")
    argParser.add_argument(
        "-l", "--labels", action="store", type=str, nargs="+",
        help="Labels to process, defaults to all valid labels")
    argParser.add_argument(
        "-c", "--channel", action="store", type=str, default="Ch2",
        help="Channel to process")
    argParser.add_argument(
        "-g", "--decay", action="store_true",
        help="Pre-compute decay time based on GCaMP6f and imaging rate")
    argParser.add_argument(
        "-r", "--raw", action="store_true",
        help="Estimate spikes from the raw fluorescence (uses dfof by default)")
    argParser.add_argument(
        "-mf", "--mad_filter", action="store", type=float, default=None,
        help="Number of MADs of original trace to use as threshold for \
              retaining spikes. '-mf 3' will give a cutoff of approximately 2 s.d. \
              under a normal distribution")
    argParser.add_argument(
        "-b", "--baseline", action="store", type=float, default=[None, None],
        nargs="*",
        help="Baseline percentile to subtract in rolling window from the signal"
             + "to remove slow changes. Enter as 2 integers, e.g.:"
             + " -b percentile num_frames. If first number is -1, minimax baseline is used")
    argParser.add_argument(
        "-s", "--savgol", action="store", type=int, default=None,
        help="Whether to subtract a rolling 30 sec average of the trace before"
             + "running spike inferencing, via a 1st order Savitsky-Golay filter")
    argParser.add_argument(
        "-re", "--refine", action="store", type=int, default=0,
        help="When using MAD filtering, whether to iteratively re-estimate "
             + "the MAD threshold by estimating the residuals through reconvolution "
             + "and subtraction of the current spike set. Parameter sets the number "
             + "of iterations")
    argParser.add_argument(
        "-R", "--retain_calcium", action="store_true",
        help="Retain denoised calcium and baseline")
    argParser.add_argument(
        "-o", "--overwrite", action="store_true",
        help="overwrite existing spikes")
    argParser.add_argument(
        "-sp", "--save_params", action="store_true",
        help="Save parameters alongside spikes")
    argParser.add_argument(
        "-sd", "--save_df", action="store_true",
        help="Save DF into dff.pkl")
    argParser.add_argument(
        "--lam", action="store", type=float, default=7,
        help="Sparsity penalty for AR1 model")

    args = argParser.parse_known_args(argv)[0]


    if args.trial_id:
        exptsToProcess = [dbExperiment(args.trial_id)]
    
    elif args.mouse is not None:
        exptsToProcess = []
        for mouse in args.mouse:
            dbmouse = dbMouse(mouse)
            exptsToProcess.extend([dbExperiment(x) for x in dbmouse._experiments])
        print("{} experiments to process".format(len(exptsToProcess)))
    else:
        'Give either Trial ID or Mouse Name'

    for expt in exptsToProcess:
        try:
            channel = expt.imaging_dataset().channel_names[
                expt.imaging_dataset()._resolve_channel(args.channel)]
        except NoSimaPath:
            print("Experiment not synchronized with imaging data. Skipping: {} ".format(expt.trial_id))
            continue

        if not expt.hasSignalsFile(channel=channel):
            print("Experiment doesn't have signals data. Skipping: {} ".format(expt.trial_id))
            continue

        labels = labelsToProcess(expt, channel=channel, labels=args.labels,
                                 raw=args.raw)
        print("Processing {} labels".format(labels))
        if args.decay:
            print("Processing decay")
            # Calculated based on equation given in ipython notebook
            im_rate = 1 / expt.frame_period()
            tau_D = 0.4
            # tau_D = 2

            g = np.exp(-1 / (tau_D * im_rate))

        else:
            g = None

        for label in labels:
            print("Processing {} label".format(label))
            overwrite = args.overwrite
            print("Overwrite: ".format(overwrite))
            if overwrite or not \
                    already_processed(expt, channel, label):
                print 'Processing {} spikes for {}:{}'.format(
                    label, expt.parent.mouseID, expt.get('startTime'))

                spikes = oasis_spikes(expt, label=label, channel=channel,
                                      g=g, raw=args.raw, savgol=args.savgol,
                                      mad_filter=args.mad_filter,
                                      baseline=args.baseline,
                                      refine=args.refine,
                                      lam=args.lam)
            else:
                print("Experiment {} already has spikes".format(expt.trial_id))
                continue

            if args.save_df:
                try:
                    with open(expt.dfofFilePath(), 'rb') as fp:
                        dff = pkl.load(fp)
                except IOError:
                    dff = {}

                with open(expt.roisFilePath(), 'rb') as f:
                    roi_list = pkl.load(f)[label]['rois']

                dff[label] = {}
                dff[label]['traces'] = spikes['signals']
                dff[label]['baseline'] = spikes['baseline']
                dff[label]['params'] = spikes['params']
                dff[label]['params'].update({'method': 'OASIS DF'})
                dff[label]['rois'] = roi_list
                with open(expt.dfofFilePath(), 'wb') as fw:
                    pkl.dump(dff, fw)

            if args.save_params:
                params = str(spikes['params'])
                plabel = label + '_params'

            if not args.retain_calcium:
                spikes = {'spikes': spikes['spikes']}

            spikes_file = expt.spikesFilePath(channel=channel)
            if os.path.splitext(spikes_file)[1] == '.h5':
                with h5py.File(spikes_file, 'a') as f:
                    if channel not in f.keys():
                        f.create_group(channel)
                    if f[channel].get(label) is not None:
                        del f[channel][label]

                    for key in spikes.keys():
                        f.create_dataset(
                            os.path.join(channel, label),
                            data=spikes[key], compression='gzip')

                    if args.save_params:
                        if f[channel].get(plabel) is not None:
                            del f[channel][plabel]
                        f.create_dataset(
                            os.path.join(channel, plabel),
                            data=params)
            else:
                try:
                    with open(spikes_file, 'rb') as fp:
                        spikes_data = pkl.load(fp)
                except (IOError, pkl.UnpicklingError):
                    spikes_data = {}
                spikes_data[label] = spikes
                with open(spikes_file, 'wb') as fp:
                    pkl.dump(spikes_data, fp)


if __name__ == '__main__':
    main(sys.argv[1:])
