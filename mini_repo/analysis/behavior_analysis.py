import numpy as np
from matplotlib import pyplot as plt
import itertools as it
from scipy.ndimage.filters import gaussian_filter1d
from scipy.interpolate import interp1d

import mini_repo.classes.exceptions as exc

def absolutePosition(trial, imageSync=True, sampling_interval=None, interp_method="linear"):
    """Returns the normalized absolute position of the mouse at each imaging time frame

    Keyword arguments:
    imageSync -- if True, syncs to imaging data

    absolutePosition % 1 = behaviorData()['treadmillPosition']

    """

    assert not (imageSync and sampling_interval is not None)

    if not imageSync and sampling_interval is None:
        raise(Exception,
            "Should be either image sync'd or at an explicit sampling " +
            "interval, defaulting to 'actual' sampling interval")
        sampling_interval = trial.behavior_sampling_interval()
        
    if(sampling_interval=="actual"):
        sampling_interval = trial.behavior_sampling_interval()
    
    if(imageSync):
        sampling_interval = trial.parent.frame_period()

    with open(trial.behaviorDataPath()) as fh:
        bd = pkl.load(fh)
    try:
        position = bd['treadmillPosition'][:, 1]
        time = bd['treadmillPosition'][:, 0]
    except KeyError:
        raise exc.MissingBehaviorData(
            'No treadmillPosition, unable to calculate absolute position')

    # if not imageSync:
    #     full_position = np.empty(
    #         int(bd['recordingDuration'] / bd['samplingInterval']))
    #     for tt, pos in position:
    #         full_position[int(tt / bd['samplingInterval']):] = pos
    #     position = full_position

    lap_starts = np.where(np.diff(position) < -0.5)[0]
    lap_back = np.where(np.diff(position) > 0.5)[0].tolist()
    lap_back.reverse()

    # Need to check for backwards steps around the lap start point
    if len(lap_back) > 0:
        next_back = lap_back.pop()
    else:
        next_back = np.inf

    for start in lap_starts:
        if next_back < start:
            position[next_back + 1:] -= 1
            position[start + 1:] += 1
            if len(lap_back) > 0:
                next_back = lap_back.pop()
            else:
                next_back = np.inf
        else:
            position[start + 1:] += 1
            
    duration = np.float(bd["recordingDuration"])
    outputTime = np.arange(0., duration, sampling_interval)
    if imageSync:
        outputTime = outputTime[:trial.parent.num_frames()]
    
    interpMethod = interp1d(time, position, 
                            kind=interp_method, bounds_error=False,
                            fill_value=position[-1])
        
    position = interpMethod(outputTime)

    return position


def velocity(trial, imageSync=True, sampling_interval=None, belt_length=200,
             smoothing=None, window_length=5, tick_count=None, interp_method='linear'):
    """Return the velocity of the mouse.

    Parameters
    ----------
    imageSync : bool
        If True, syncs to imaging data.
    belt_length : float
        Length of belt, will return velocity in units/second.
    smoothing {None, str}
        Window function to use, should be 'flat' for a moving average or
        np.'smoothing' (hamming, hanning, bartltett, etc.).
    window_length int
        Length of window function, should probably be odd.
    tick_count : float
        if not None velocity is calculated based on the treadmill
        times by counting ticks and dividing by the tick_count. i.e.
        tick _count should be in ticks/m (or ticks/cm) to get m/s (cm/s)
        returned.

    """
    assert not (imageSync and sampling_interval is not None)

    if not imageSync and sampling_interval is None:
        warnings.warn(
            "Should be either image sync'd or at an explicit sampling " +
            "interval, defaulting to 'actual' sampling interval")
        sampling_interval = 'actual'

    try:
        b = trial.parent.belt().length()
        assert b > 0
        belt_length = b
    except (exc.NoBeltInfo, AssertionError):
        try:
            belt_length = trial.parent.track_length/10.0
        except:
            warnings.warn('No belt information found for experiment %s.  \nUsing default belt length = %f' % (str(trial.parent), belt_length))

    if tick_count is not None:
        bd = trial.behaviorData(imageSync=imageSync)
        times = bd['treadmillTimes']
        duration = bd['recordingDuration']
        if imageSync:
            times = np.where(times != 0)[0] * trial.parent.frame_period()
            bincounts = np.bincount(
                times.astype(int), minlength=duration)[:duration]
            bincounts = bincounts.astype(float) / tick_count
            interpFunc = scipy.interpolate.interp1d(
                range(len(bincounts)), bincounts)
            xnew = np.linspace(
                0, len(bincounts) - 1, len(bd['treadmillTimes']))
            return interpFunc(xnew)
        else:
            bincounts = np.bincount(
                times.astype(int), minlength=duration)[:duration]
            bincounts = bincounts.astype(float) / tick_count
            return bincounts

    try:
        position = absolutePosition(
            trial, imageSync=imageSync, sampling_interval=sampling_interval, interp_method=interp_method)
    except exc.MissingBehaviorData:
        raise exc.MissingBehaviorData(
            'Unable to calculate position based velocity')

    if imageSync:
        samp_int = trial.parent.frame_period()
    elif sampling_interval == 'actual':
        samp_int = trial.behavior_sampling_interval()
    else:
        samp_int = sampling_interval

    vel = np.hstack([0, np.diff(position)]) * belt_length / samp_int

    if smoothing is not None and np.any(vel != 0):
        if smoothing == 'flat':  # moving average
            w = np.ones(window_length, 'd')
        else:
            # If 'smoothing' is not a valid method this will throw an AttributeError
            w = eval('np.' + smoothing + '(window_length)')
        s = np.r_[vel[window_length - 1:0:-1], vel, vel[-1:-window_length:-1]]
        vel = np.convolve(w / w.sum(), s, mode='valid')
        # Trim away extra frames
        vel = vel[window_length / 2 - 1:-window_length / 2]

    return vel