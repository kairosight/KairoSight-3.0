# from util.preparation import * FIGURE OUT HOW TO MAKE THIS WORK!!!

import statistics
import sys

import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyval
from scipy.interpolate import LSQUnivariateSpline
from scipy.signal import find_peaks, correlate, filtfilt, kaiserord, firwin,\
    butter, convolve2d, remez
from scipy.optimize import curve_fit
from skimage.morphology import square
from skimage.filters import gaussian
from skimage.filters.rank import median, mean, mean_bilateral

# Constants
# LSQ Spline fidelity
SPLINE_FIDELITY = 3
# Baseline sample number limits
BASELINES_MIN = 5
BASELINES_MAX = 20
# Transient Signal-to-Noise limit
SNR_MIN = 5.0
SNR_MAX = 100
# Baseline sample number limits
FILTERS_SPATIAL = ['median', 'mean', 'bilateral', 'gaussian', 'best_ever']


# TODO add TV, a non-local, and a weird filter


def spline_signal(signal_in):
    xx_signal = np.arange(0, (len(signal_in)))
    # Lease Square approximation
    # Computing the inner knots and using them:
    x_spline = np.linspace(xx_signal[0], xx_signal[-1], len(xx_signal) * SPLINE_FIDELITY)
    n_knots = 35  # number of knots to use in LSQ spline
    t_knots = np.linspace(xx_signal[0], xx_signal[-1], n_knots)  # equally spaced knots in the interval
    t_knots = t_knots[2:-2]  # discard edge knots
    # t_knots = [0, 1, 2, 3]
    bspline_degree = 3
    # sql = make_lsq_spline(xx_signal, signal_in, t_knots)
    spline = LSQUnivariateSpline(xx_signal, signal_in, t_knots, k=bspline_degree)
    # x_signal = np.arange(len(signal_in))
    # x_spline = np.linspace(x_signal[0], x_signal[-1], len(x_signal) * SPLINE_FIDELITY)
    # # n_segments = int(len(x_signal) / 5)
    # n_segments = 25
    # n_knots = n_segments
    # knots = np.linspace(x_signal[0], x_signal[-1], n_knots + 2)[1:-2]
    # bspline_degree = 3
    # spline = LSQUnivariateSpline(x_signal, signal_in, knots, k=bspline_degree)
    return x_spline, spline

    # return xs, sql


def spline_deriv(signal_in):
    xs, sql = spline_signal(signal_in)

    x_df = xs
    df_spline = sql.derivative()(xs)

    return x_df, df_spline


def find_tran_peak(signal_in, props=False):
    """Find the index of the peak of a transient,
    defined as the maximum value

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float
        props : bool # TODO remove and propagate change to always return props
            Whether to return properties of the peaks, default : False

        Returns
        -------
        i_peak : np.int64
            The index of the signal array corresponding to the peak of the transient
            or NaN if no peak was detected
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    # if signal_in.dtype not in [np.uint16, float]:
    #     raise TypeError('Signal values must either be "int" or "float"')

    # if any(v < 0 for v in signal_in):
    #     raise ValueError('All signal values must be >= 0')

    # Characterize the signal
    unique, counts = np.unique(signal_in, return_counts=True)
    if len(unique) < 10:  # signal is too flat to have a valid peak
        if props:
            return np.nan, np.nan
        else:
            return np.nan

    # Replace NaNs with 0
    # signal_in = np.nan_to_num(signal_in, copy=False, nan=0)

    signal_bounds = (signal_in.min(), signal_in.max())
    signal_mean = np.nanmean(signal_in)
    if signal_in.dtype is np.uint16:
        signal_mean = int(np.floor(np.nanmean(signal_in)))
    signal_range = signal_bounds[1] - signal_mean

    # TODO detect dual peaks, alternans, etc.

    # # Roughly find the peaks using a smoothing wavelet transformation
    # distance = int(len(signal_in) / 2)
    # i_peaks = find_peaks_cwt(signal_in, widths=np.arange(10, distance))
    # if len(i_peaks) is 0:  # no peak detected
    #     return np.nan
    # if len(i_peaks) > 1:
    #     return i_peaks[0]
    # return i_peaks

    # Roughly find the "prominent" peaks a minimum distance from eachother
    prominence = signal_range * 0.8
    distance = int(len(signal_in) / 2)
    i_peaks, properties = find_peaks(signal_in,
                                     height=signal_mean, prominence=prominence,
                                     distance=distance)
    if len(i_peaks) == 0:  # no peak detected
        if props:
            return np.nan, np.nan
        else:
            return np.nan

    # Use the peak with the max prominence (in case of a tie, first is chosen)
    i_peak = i_peaks[np.argmax(properties['prominences'])]
    if props:
        return i_peak, properties
    else:
        return i_peak


def find_tran_baselines(signal_in, peak_side='left'):
    # Characterize the signal
    # signal_bounds = (signal_in.min(), signal_in.max())
    # signal_range = signal_bounds[1] - signal_bounds[0]
    # find the peak (roughly)
    signal_range = signal_in.max() - signal_in.min()
    i_peak = find_tran_peak(signal_in)
    if i_peak is np.nan:
        return np.nan
    signal_cutoff = signal_in.min() + (signal_range / 2)
    # i_signal_cutoff_left = np.where(signal_in[:i_peak] <= signal_cutoff)[0][0]
    i_signal_cutoff_right = np.where(signal_in[:i_peak] <= signal_cutoff)[0][-1]

    # Exclude signals without a prominent peak

    # use the derivative spline to find relatively quiescent baseline period
    xdf, df_spline = spline_deriv(signal_in)

    # TODO catch atrial-type signals and limit to the plataea before the peak
    # find the df max before the signal's peak (~ large rise time)
    df_search_left = SPLINE_FIDELITY * SPLINE_FIDELITY

    # include indexes within the standard deviation of the local area of the derivative
    df_sd = statistics.stdev(df_spline[df_search_left:-df_search_left])
    df_prominence_cutoff = df_sd * 2

    df_max_search_right = i_signal_cutoff_right * SPLINE_FIDELITY

    i_peak_df = df_search_left + np.argmax(df_spline[df_search_left:df_max_search_right])
    df_search_start_right = i_peak_df

    # i_min_df = df_search_left + np.argmin(df_spline[df_search_left:df_search_start_right])
    i_start_df = i_peak_df

    # find first value within cutoff
    df_spline_search = df_spline[:i_peak_df+1]
    for idx_flip, value in enumerate(np.flip(df_spline_search)):
        if abs(value) < df_prominence_cutoff:
            i_start_df = i_peak_df - idx_flip
            break

    i_left_df = i_start_df
    i_right_df = i_start_df
    # look left TODO allow to go further (higher cutoff?) to not overestimate noisy SNRs
    for value in np.flip(df_spline[df_search_left:i_start_df]):
        if abs(value) < df_prominence_cutoff:
            i_left_df = i_left_df - 1
        else:
            break
    # look right
    for value in df_spline[i_start_df:i_peak_df]:
        if abs(value) < df_prominence_cutoff:
            i_right_df = i_right_df + 1
        else:
            break
    # combine
    i_baselines_search = np.arange(i_left_df, i_right_df)

    if (i_right_df > i_peak_df) or (len(i_baselines_search) < (BASELINES_MIN * SPLINE_FIDELITY)):
        print('\n\t\t* df_cutoff: {} gives [{}:{}]\ti_start_df[{}]: {}\tfrom i_peak_df[{}]: {}'
              .format(round(df_prominence_cutoff, 3), i_left_df, i_right_df,
                      i_start_df, round(df_spline[i_start_df], 3),
                      i_peak_df, round(df_spline[i_peak_df], 3)))

        if i_right_df > i_peak_df:
            return np.nan

        # use arbitrary backup baselines: the 10 signal samples before the df search start (non-inclusive)
        i_right_df = int(i_right_df / SPLINE_FIDELITY)
        if i_right_df > BASELINES_MIN:
            i_baselines_backup = np.arange(i_right_df - BASELINES_MIN, i_right_df)
        else:
            i_baselines_backup = np.arange(0, BASELINES_MIN)
        return i_baselines_backup

    # use all detected indexes
    i_baselines_left = int(i_baselines_search[0] / SPLINE_FIDELITY)
    i_baselines_right = int(i_baselines_search[-1] / SPLINE_FIDELITY)

    i_baselines = np.arange(i_baselines_left, i_baselines_right)
    if len(i_baselines) > BASELINES_MAX:
        i_baselines = i_baselines[-BASELINES_MAX:]

    return i_baselines


def find_tran_act(signal_in):
    """Find the time of the activation of a transient,
    defined as the the maximum of the 1st derivative OR

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float

        Returns
        -------
        i_activation : np.int64
            The index of the signal array corresponding to the activation of the transient
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    # if signal_in.dtype not in [np.uint16, float]:
    #     raise TypeError('Signal values must either be "int" or "float"')

    # Limit the search to be well before
    # and well after the peak (depends on which side of the peak the baselines are)
    i_peak = find_tran_peak(signal_in)
    if i_peak is np.nan:
        return np.nan
    i_baselines = find_tran_baselines(signal_in)
    # i_baseline = int(np.median(i_baselines))
    if i_baselines is np.nan:
        return np.nan

    baselines_rms = np.sqrt(np.mean(signal_in[i_baselines]) ** 2)
    peak_peak = signal_in[i_peak] - baselines_rms
    data_noise = signal_in[i_baselines]
    noise_sd = statistics.stdev(data_noise.astype(float))  # standard deviation
    snr = peak_peak / noise_sd
    if snr < SNR_MIN:
        print('\t ** SNR too low to analyze: {}'.format(round(snr, 3)))
        return np.nan

    search_min = i_baselines[-1]  # TODO try the last baseline index
    search_max = i_peak

    # use a LSQ derivative spline of entire signal
    x_df, signal_df = spline_deriv(signal_in)

    # find the 1st derivative max within the search area (first few are likely to be extreme)
    i_act_search_df = np.argmax(signal_df[search_min * SPLINE_FIDELITY:search_max * SPLINE_FIDELITY])
    i_act_search = int(np.floor(i_act_search_df / SPLINE_FIDELITY))

    i_activation = search_min + i_act_search

    if i_activation == i_peak:
        print('\tWarning! Activation time same as Peak: {}'.format(i_activation))

    return i_activation


def align_signals(signal1, signal2):
    """Aligns two signal arrays using signal.correlate.
    https://stackoverflow.com/questions/19642443/use-of-pandas-shift-to-align-datasets-based-on-scipy-signal-correlate

        Parameters
        ----------
        signal1 : ndarray, dtype : uint16 or float
            Signal array
        signal2 : ndarray, dtype : uint16 or float
            Signal array, will be aligned to signal1

        Returns
        -------
        signal2_aligned : ndarray
            Aligned version of signal2
        shift : int
            Number of indexes signal2 was shifted during alignment

        Notes
        -----
            Signal arrays must be the same length?
            Should not be applied to signal data containing at least one transient.
            Fills empty values with np.NaN
    """
    # Set signal datatype as float32
    sig1 = np.float32(signal1)
    sig2 = np.float32(signal2)

    # Find the length of the signal
    # sig_length = len(sig1)
    # print('sig1 min, max: ', np.nanmin(sig1), ' , ', np.nanmax(sig1))
    # print('sig2 min, max: ', np.nanmin(sig2), ' , ', np.nanmax(sig2))

    # dx = np.mean(np.diff(sig1.x.values))
    shift = (np.argmax(correlate(sig1, sig2)) - len(sig2))

    signal2_aligned = np.roll(sig2, shift=shift+1)
    if shift > 0:
        signal2_aligned[:shift] = np.nan
    else:
        signal2_aligned[shift:] = np.nan

    return signal2_aligned, shift


def isolate_spatial(stack_in, roi):
    """Isolate a spatial region of a stack (3-D array, TYX) of grayscale optical data.

        Parameters
        ----------
        stack_in : ndarray, dtype : uint16 or float
             A 3-D array (T, Y, X) of optical data
        roi : `GraphicsItem <pyqtgraph.graphicsItems.ROI>`
             Generic region-of-interest widget.

        Returns
        -------
        stack_out : ndarray
             A spatially isolated 3-D array (T, Y, X) of optical data, dtype : stack_in.dtype
       """
    pass


def isolate_temporal(stack_in, i_start, i_end):
    """Isolate a temporal region of a stack (3-D array, TYX) of grayscale optical data.

        Parameters
        ----------
        stack_in : ndarray
             A 3-D array (T, Y, X) of optical data, dtype : uint16 or float
        i_start : int
             Index or frame to start temporal isolation
        i_end : int
             Index or frame to end temporal isolation

        Returns
        -------
        stack_out : ndarray
             A temporally isolated 3-D array (T, Y, X) of optical data, dtype : stack_in.dtype
        """
    pass


def isolate_transients(signal_in, i_start=0, i_end=None):
    """Isolate similar transients from a signal array of optical data.
        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float
        i_start : int, optional
            Index or frame to start transient isolation. The default is 0
        i_end : int, optional
            Index or frame to end transient isolation. The default is None.


        Returns
        -------
        transients : list
            The isolated arrays of transient data, dtype : signal_in.dtype
        cycle : int
            The estimated cycle length, indexes between transient peaks TODO change to activation times
        """
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "uint16" or "float"')

    # Calculate the number of transients in the signal
    if i_end:
        signal_in = signal_in[:i_end]
    # Characterize the signal
    unique, counts = np.unique(signal_in, return_counts=True)

    if len(unique) < 10:  # signal is too flat to have a valid peak
        return np.zeros_like(signal_in)

    # Find the peaks
    # i_peaks, _ = find_peaks(signal_in)
    signal_bounds = (signal_in.min(), signal_in.max())
    signal_mean = np.nanmean(signal_in)
    if signal_in.dtype is np.uint16:
        signal_mean = int(np.floor(np.nanmean(signal_in)))
    signal_range = signal_bounds[1] - signal_mean
    prominence = signal_range * 0.8
    distance_min = 10
    i_peaks, properties = find_peaks(signal_in,
                                     height=signal_mean, prominence=prominence,
                                     distance=distance_min)

    if len(i_peaks) == 0:
        raise ArithmeticError('No peaks detected'.format(len(i_peaks), i_peaks))
    if len(i_peaks) > 3:
        print('* {} peaks detected at {} in signal_in'.format(len(i_peaks), i_peaks))
    else:
        raise ValueError('Only {} peak detected at {} in signal_in'.format(len(i_peaks), i_peaks))

    # do not use the first and last peaks
    i_peaks = i_peaks[1:-1]
    # Split up the signal using peaks and estimated cycle length
    est_cycle_array = np.diff(i_peaks).astype(float)
    cycle = int(np.floor(np.nanmean(est_cycle_array)))
    cycle_shift = int(np.floor(cycle / 2))

    signals_trans_peak = []
    i_baselines_full = []
    i_acts_full = []
    signals_trans_act = []

    # roughly isolate all transients centered on their peaks
    # and cropped with a cycle-length-wide window
    for peak_num, peak in enumerate(i_peaks):
        sig = signal_in[i_peaks[peak_num] - cycle_shift:
                        i_peaks[peak_num] + cycle_shift]
        signals_trans_peak.append(sig)

        i_baselines = find_tran_baselines(sig)
        i_baselines_full.append((i_peaks[peak_num] - cycle_shift) + i_baselines)
        i_act_signal = find_tran_act(sig)
        i_act_full = (i_peaks[peak_num] - cycle_shift) + i_act_signal
        i_acts_full.append(i_act_full)

    # TODO exclude those with: abnormal rise times, low OWS ...

    # With that peak detection, find activation times and align transient
    shift_max = 0
    for act_num, i_act_full in enumerate(i_acts_full):
        # if crop is 'center':
        # center : crop transients using the cycle length
        # cropped to center at the alignment points

        # align along activation times, and crop to include ensuing diastolic intervals
        i_t_start = i_act_full - (i_acts_full[0] - i_baselines_full[0][0])
        i_t_end = i_act_full + (i_peaks[1] - i_acts_full[0])

        signal_align = signal_in[i_t_start:i_t_end]

        signal_align = normalize_signal(signal_align)
        # Use correlation to align the transients
        shift_max = 0
        if act_num > 0:
            signal_align, shift = align_signals(signals_trans_act[0], signal_align)
            shift_max = np.nanmax([shift_max, abs(shift)])
        signals_trans_act.append(signal_align)

    transients = [sig[:-shift_max] for sig in signals_trans_act]

    return transients, cycle


def filter_spatial(frame_in, filter_type='gaussian', kernel=3):
    """Spatially filter a frame (2-D array, YX) of grayscale optical data.

        Parameters
        ----------
        frame_in : ndarray
             A 2-D array (Y, X) of optical data, dtype : uint16 or float
        filter_type : str
            The type of filter algorithm to use, default is gaussian
        kernel : int
            The width and height of the kernel used, must be positive and odd, default is 3

        Returns
        -------
        frame_out : ndarray
             A spatially filtered 2-D array (Y, X) of optical data, dtype : frame_in.dtype
        """
    # Check parameters
    if type(frame_in) is not np.ndarray:
        raise TypeError('Frame type must be an "ndarray"')
    if len(frame_in.shape) != 2:
        raise TypeError('Frame must be a 2-D ndarray (Y, X)')
    if frame_in.dtype not in [np.uint16, float]:
        raise TypeError('Frame values must either be "np.uint16" or "float"')
    if type(filter_type) is not str:
        raise TypeError('Filter type must be a "str"')
    if type(kernel) is not int:
        raise TypeError('Kernel size must be an "int"')

    if filter_type not in FILTERS_SPATIAL:
        raise ValueError('Filter type must be one of the following: {}'.format(FILTERS_SPATIAL))
    if kernel < 3 or (kernel % 2) == 0:
        raise ValueError('Kernel size {} px must be >= 3 and odd'.format(kernel))

    if filter_type == 'median':
        # Good for ___, but ___
        # k = np.full([kernel, kernel], 1)
        frame_out = median(frame_in, square(kernel))
    elif filter_type == 'mean':
        # Good for ___, but over-smooths?
        # k = np.full([kernel, kernel], 1)
        frame_out = mean(frame_in, square(kernel))
    elif filter_type == 'bilateral':
        # Good for edge preservation, but slow
        # sigma_color = 50  # standard deviation of the intensity gaussian kernel
        # sigma_space = 10  # standard deviation of the spatial gaussian kernel
        frame_out = mean_bilateral(frame_in, square(kernel))
    elif filter_type == 'gaussian':
        # Good for ___, but ___
        sigma = kernel  # standard deviation of the gaussian kernel
        frame_out = gaussian(frame_in, sigma=sigma, mode='mirror', preserve_range=True)
    else:
        raise NotImplementedError('Filter type "{}" not implemented'.format(filter_type))

    return frame_out.astype(frame_in.dtype)


def filter_spatial_stack(stack_in, kernel_size=3):
    """Spatially filter a stack (3-D array, YXZ) of grayscale optical data.

        Parameters
        ----------
        stack_in : ndarray
             A 3-D array (Y, X, Z) of optical data, dtype : uint16 or float
        kernel_size : int
            The width and height of the kernel used, must be positive and odd,
            default is 3, max is 7

        Returns
        -------
        stack_out : ndarray
             A spatially filtered 3-D array (Y, X) of optical data,
             dtype : frame_in.dtype
        """
    # Create the kernel
    pattern = np.ones([kernel_size, kernel_size])
    # Create a mask to remove background data from consideration
    mask = stack_in[0, :, :] == 0
    # Preallocate output variable
    stack_out = np.zeros(stack_in.shape)
    # Iterate through the data
    for c, data in enumerate(stack_in):
        # Use 2D convolution to calculate the average
        ave = convolve2d(np.where(mask, 0, data), pattern, 'same')\
            / convolve2d(~mask, pattern, 'same')
        ave[mask] = 0
        stack_out[c, :, :] = ave
    # Return the average stack
    return stack_out

def filter_temporal(signal_in, sample_rate, filter_order, freq_cutoff):
    """Apply a lowpass filter to an array of optical data.

        Parameters
        ----------
        signal_in : ndarray
             The array of data to be evaluated, dtype : uint16 or float
        sample_rate : float
            Sample rate (Hz) of signal_in
        freq_cutoff : float
            Cutoff frequency (Hz) of the lowpass filter, default is 100
        filter_order : int or str
            The order of the filter, default is 'auto'
            If 'auto', order is calculated using scipy.signal.kaiserord

        Returns
        -------
        signal_out : ndarray
             A temporally filtered signal array, dtype : signal_in.dtype
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "uint16" or "float"')
    if type(sample_rate) is not float:
        raise TypeError('Sample rate must be a "float"')
    if type(freq_cutoff) is not float:
        raise TypeError('Cutoff frequency must be a "float"')
    if type(filter_order) not in [int, str]:
        raise TypeError('Filter type must be an int or str')

    nyq_rate = sample_rate / 2.0
    n_order = 0
    if type(filter_order) is int:
        # Calculate filter coefficients using Remez exchange algorithm
        b = remez(filter_order, [0.5, freq_cutoff, freq_cutoff*1.1,
                                 sample_rate/2.0], [1, 0], Hz=sample_rate)
        a = 1.0
        # Reshape the data to 2D array with signals in the columns
        data = np.reshape(signal_in, (signal_in.shape[0],
                                      signal_in.shape[1]*signal_in.shape[2]))
        # Apply the filter to all the data at once
        try:
            data = filtfilt(b, a, data, axis=0)
        except ValueError:
            print('Your data is of insufficient length for standard temporal filtering.')
            print('The padding has been reduced accordingly.')
            print('Trying again.')
            data = filtfilt(b, a, data, axis=0, padlen=data.shape[0]-1)
        # Return data to original shape
        signal_out = np.reshape(data, (signal_in.shape[0],
                                       signal_in.shape[1], signal_in.shape[2]))
    elif filter_order == 'auto':
        # # FIR 4 design  -
        # https://www.programcreek.com/python/example/100540/scipy.signal.firwin
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
        # Compute the order and Kaiser parameter for the FIR filter.
        ripple_db = 30.0
        width = 20  # The desired width of the transition from pass to stop, Hz
        window = 'kaiser'
        n_order, beta = kaiserord(ripple_db, width / nyq_rate)
        # Use firwin with a Kaiser window to create a lowpass FIR filter.
        taps = firwin(numtaps=n_order + 1, cutoff=freq_cutoff,
                      window=(window, beta), fs=sample_rate)
        print(taps)
        # signal_out = lfilter(taps, 1.0, signal_in)   # for FIR, a=1
        # for FIR, a=1
        signal_out = filtfilt(taps, 1, signal_in, method="gust")
        # # Savitzky Golay
        # window_coef = int(nyq_rate / 50)
        #
        # if window_coef % 2 > 0:
        #     window = window_coef
        # else:
        #     window = window_coef + 1
        #
        # signal_out = savgol_filter(signal_in, window, 3)
    else:
        raise ValueError(
            'Filter order "{}" not implemented'.format(filter_order))
    return signal_out.astype(signal_in.dtype)


def filter_drift(signal_in, mask, drift_order):
    """Remove drift from an array of optical data using the subtraction of a
    polynomial fit.

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float
        drift_order : int or str
            The order of the polynomial drift to fit to, default is 'exp'
            If 'exp', drift is calculated using scipy.optimize.curve_fit

        Returns
        -------
        signal_out : ndarray
            A signal array with drift removed, dtype : signal_in.dtype
        drift : ndarray
            The values of the calculated polynomial drift used
    """
    # Clone the signal
    signal_out = signal_in
    # Iterate through the data removing drift from all non-masked out pixels
    for n in np.arange(0, signal_in.shape[1]):
        for m in np.arange(0, signal_in.shape[2]):
            if mask[n, m]:
                # Create vector of x-values along which to fit the curve
                sig_x = np.arange(0, len(signal_in[:, n, m]))
                # Least squares polynomial fit of specified order
                z = Polynomial.fit(sig_x, signal_in[:, n, m], drift_order)
                '''# Conversion of fit to a polynomial class
                p = np.poly1d(z)'''
                # Evaluation of fit polynomial
                drift_rm = polyval(sig_x, z.convert().coef)
                # Remove drift
                signal_out[:, n, m] = signal_in[:, n, m]-drift_rm
    # Return the results
    return signal_out

def filter_drift2(signal_in, drift_order):
    """Remove drift from an array of optical data using the subtraction of a
    polynomial fit.

        Parameters
        ----------
        signal_in : ndarray
            The array of data to be evaluated, dtype : uint16 or float
        drift_order : int or str
            The order of the polynomial drift to fit to, default is 'exp'
            If 'exp', drift is calculated using scipy.optimize.curve_fit

        Returns
        -------
        signal_out : ndarray
            A signal array with drift removed, dtype : signal_in.dtype
        drift : ndarray
            The values of the calculated polynomial drift used
    """
    # Clone the signal
    signal_out = signal_in
    # Iterate through the data removing drift from all non-masked out pixels
    for n in np.arange(0, signal_in.shape[1]):
        for m in np.arange(0, signal_in.shape[2]):
            #if mask[n, m]:
                # Create vector of x-values along which to fit the curve
                sig_x = np.arange(0, len(signal_in[:, n, m]))
                # Least squares polynomial fit of specified order
                z = Polynomial.fit(sig_x, signal_in[:, n, m], drift_order)
                '''# Conversion of fit to a polynomial class
                p = np.poly1d(z)'''
                # Evaluation of fit polynomial
                drift_rm = polyval(sig_x, z.convert().coef)
                # Remove drift
                signal_out[:, n, m] = signal_in[:, n, m]-drift_rm
    # Return the results
    return signal_out



def invert_signal(signal_in):
    """Invert the values of a signal array.

        Parameters
        ----------
        signal_in : ndarray
             The array of data to be processed, dtype : uint16 or float

        Returns
        -------
        signal_out : ndarray
             The inverted signal array, dtype : signal_in.dtype
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "uint16" or "float"')

    unique, counts = np.unique(signal_in, return_counts=True)
    if len(unique) == 1:
        return signal_in

    # calculate axis to rotate data around (middle value int or float)
    axis = signal_in.min() + ((signal_in.max() - signal_in.min()) / 2)
    if signal_in.dtype in [np.int32]:
        axis = np.floor(axis).astype(int)

    # rotate the data around it's central value
    signal_out = (axis + (axis - signal_in)).astype(signal_in.dtype)

    return signal_out


def invert_stack(stack_in):
    """Invert the values of an image stack (3-D array).

        Parameters
        ----------
        stack_in : ndarray
            Image stack with shape (T, Y, X)

        Returns
        -------
        stack_out : ndarray
            A cropped 3-D array (T, Y, X) of optical data, dtype : float
        """
    # Check parameters
    if type(stack_in) is not np.ndarray:
        raise TypeError('Stack type must be an "ndarray"')
    if len(stack_in.shape) != 3:
        raise TypeError('Stack must be a 3-D ndarray (T, Y, X)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')

    stack_out = np.empty_like(stack_in)
    map_shape = stack_in.shape[1:]
    # Assign a value to each pixel
    for iy, ix in np.ndindex(map_shape):
        print('\r\tInversion of Row:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(
            iy + 1, map_shape[0], ix + 1, map_shape[1]),
              end='', flush=True)
        pixel_data = stack_in[:, iy, ix]
        pixel_data_inv = invert_signal(pixel_data)
        stack_out[:, iy, ix] = pixel_data_inv

    return stack_out


def normalize_signal(signal_in):
    """Normalize the values of a signal array to range from 0 to 1.

        Parameters
        ----------
        signal_in : ndarray
             The array of data to be processed, dtype : uint16 or float

        Returns
        -------
        signal_out : ndarray
             The normalized signal array, dtype : float
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    # if signal_in.dtype not in [np.uint16, float]:
    #     raise TypeError('Signal values must either be "uint16" or "float"')

    # if any(v < 0 for v in signal_in):
    #     raise ValueError('All signal values must be >= 0')

    unique, counts = np.unique(signal_in, return_counts=True)

    if len(unique) < 10:  # signal is too flat to have a valid peak
        return np.zeros_like(signal_in)

    xp = [signal_in.min(), signal_in.max()]
    fp = [0, 1]
    signal_out = np.interp(signal_in, xp, fp)

    return signal_out


def normalize_stack(stack_in):
    """Normalize the values of an image stack (3-D array) to range from 0 to 1.

        Parameters
        ----------
        stack_in : ndarray
            Image stack with shape (T, Y, X), dtype : uint16 or float

        Returns
        -------
        stack_out : ndarray
            A normalized image stack (T, Y, X), dtype : float
        """
    # Check parameters
    if type(stack_in) is not np.ndarray:
        raise TypeError('Stack type must be an "ndarray"')
    if len(stack_in.shape) != 3:
        raise TypeError('Stack must be a 3-D ndarray (T, Y, X)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')

    stack_out = np.empty_like(stack_in, dtype=float)
    map_shape = stack_in.shape[1:]
    # Assign a value to each pixel
    for iy, ix in np.ndindex(map_shape):
        pixel_data = stack_in[:, iy, ix]
        pixel_data_norm = normalize_signal(pixel_data)
        stack_out[:, iy, ix] = pixel_data_norm

    return stack_out


def calc_ff0(signal_in):
    """Normalize a fluorescence signal against a resting fluorescence,
    i.e. F_t / F0

        Parameters
        ----------
        signal_in : ndarray
            The array of fluorescent data (F_t) to be normalized, dtype : uint16 or float
        Returns
        -------
        signal_out : ndarray
            The array of F/F0 fluorescence data, dtype : float

        Notes
        -----
            Should not be applied to normalized or drift-removed data.
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "int" or "float"')

    # F / F0: (F_t - F0) / F0
    f_t = signal_in
    f_0 = signal_in.min()

    signal_out = (f_t - f_0) / f_0

    return signal_out


def calculate_snr(signal_in, noise_count=10):
    """Calculate the Signal-to-Noise ratio of a signal array,
    defined as the ratio of the Peak-Peak amplitude to the sample standard deviation of the noise.

        Parameters
        ----------
        signal_in : ndarray
             The array of data to be evaluated, dtyoe : int or float
        noise_count : int
             The number of noise values to be used in the calculation, default is 10

        Returns
        -------
        snr : float
             The Signal-to-Noise ratio of the given data, recommend using round(snr, 5)
        rms_bounds : tuple
             The RMSs of the peak and noise arrays, (noise_rms, peak_rms)
        peak_peak : float
             The absolute difference between the RMSs of the peak and noise arrays
        ratio_noise : float
             The ratio between the ranges of noise and peak value(s)
        sd_noise : float
             The standard deviation of the noise values
        ir_noise : ndarray
             The indexes of noise values used in the calculation
        ir_peak : int
             The index of peak values used in the calculation

        Notes
        -----
            Must be applied to signals with upward deflections (Peak > noise)
            Assumes noise SD > 1, otherwise set to 0.5
            Assumes max noise value < (peak / 5)
            Auto-detects noise section as the last noise_count values before the final noisy peak
        """
    # Check parameters
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if type(noise_count) is not int:
        raise TypeError('Number of noise values to use must be an "int"')
    if noise_count < 0:
        raise ValueError('Noise count must be >= 0')
    if noise_count >= len(signal_in):
        raise ValueError('Number of noise values to use must be < length of signal array')

    # Find peak values
    i_peak, properties = find_tran_peak(signal_in, props=True)
    if i_peak is np.nan:
        # raise ArithmeticError('No peaks detected'.format(len(i_peaks), i_peaks))
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Characterize the signal
    signal_bounds = (signal_in.min(), signal_in.max())

    i_peak_calc = i_peak
    ir_peak = i_peak_calc

    # Use the peak value
    peak_value = signal_in[i_peak_calc]

    # Find noise values
    i_noise_calc = find_tran_baselines(signal_in)

    if i_noise_calc is np.nan or len(i_noise_calc) < 5:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    data_noise = signal_in[i_noise_calc]

    # Use noise values and their RMS
    noise_rms = np.sqrt(np.mean(data_noise) ** 2)
    noise_sd = statistics.stdev(data_noise.astype(float))  # standard deviation

    # Calculate Peak-Peak value
    peak_peak = abs(peak_value - noise_rms)

    # Exclusions
    if noise_sd == 0:
        noise_sd = peak_peak / 200  # Noise data too flat to detect SD
        print('\tFound noise with SD of 0! Used {} to give max SNR of 200'.format(noise_sd))

    if signal_bounds[1] < noise_rms:
        raise ValueError('Signal max {} seems to be < noise rms {}'.format(signal_bounds[1], noise_rms))

    # Calculate SNR
    snr = peak_peak / noise_sd

    rms_bounds = (noise_rms.astype(signal_in.dtype), peak_value.astype(signal_in.dtype))
    sd_noise = noise_sd
    ir_noise = i_noise_calc

    return snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak


def map_snr(stack_in, noise_count=10):
    """Generate a map_out of Signal-to-Noise ratios for signal arrays within a stack,
    defined as the ratio of the Peak-Peak amplitude to the population standard deviation of the noise.

        Parameters
        ----------
        stack_in : ndarray
            A 3-D array (T, Y, X) of optical data, dtype : uint16 or float
        noise_count : int
             The number of noise values to be used in the calculation, default is 10

        Returns
        -------
        map : ndarray
             A 2-D array of Signal-to-Noise ratios, dtype : float

        Notes
        -----
            Pixels with incalculable SNRs assigned a value of NaN
        """
    # Check parameters
    if type(stack_in) is not np.ndarray:
        raise TypeError('Stack type must be an "ndarray"')
    if len(stack_in.shape) != 3:
        raise TypeError('Stack must be a 3-D ndarray (T, Y, X)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')

    if type(noise_count) is not int:
        raise TypeError('Noise count must be an "int"')

    # print('Generating SNR map ...')
    map_shape = stack_in.shape[1:]
    map_out = np.empty(map_shape)
    # Assign an SNR to each pixel
    for iy, ix in np.ndindex(map_shape):
        # print('\r\tRow:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix + 1, map_shape[1]),
        #       end='', flush=True)
        pixel_data = stack_in[:, iy, ix]
        # # Characterize the signal
        # signal_bounds = (pixel_data.min(), pixel_data.max())
        # signal_range = signal_bounds[1] - signal_bounds[0]

        snr, rms_bounds, peak_peak, sd_noise, ir_noise, ir_peak = calculate_snr(pixel_data, noise_count)
        # Set every pixel's values to the SNR of the signal at that pixel
        map_out[iy, ix] = snr

    # print('\nDONE Mapping SNR')
    return map_out


def calc_ensemble(time_in, signal_in, crop='center'):
    """Convert a signal from multiple transients to an averaged signal,
    segmented by activation times. Discards the first and last transients.

        Parameters
        ----------
        time_in : ndarray
            The array of timestamps (ms) corresponding to signal_in, dtyoe : int or float
        signal_in : ndarray
            The array of fluorescent data to be converted
        crop : str or tuple of ints
            The type of cropping applied, default is center
            If a tuple, begin aligned crop at crop[0] time index and end at crop[1]

        Returns
        -------
        signal_time : ndarray
            An array of timestamps (ms) corresponding to signal_out
        signal_out : ndarray
            The array of an ensembled transient signal, dtype : float
            or zeroes if no peak was detected
        signals : list
            The list of signal arrays used to create the ensemble
        i_peaks : ndarray
            The indexes of peaks from signal_in used
        i_acts  : ndarray
            The indexes of activations from signal_in used
        est_cycle : float
            Estimated cycle length (ms) of transients in signal_in

        Notes
        -----
            # Normalizes signal from 0-1 in the process
        """
    # Check parameters
    if type(time_in) is not np.ndarray:
        raise TypeError('Time data type must be an "ndarray"')
    if time_in.dtype not in [int, float]:
        raise TypeError('Time values must either be "int" or "float"')
    if type(signal_in) is not np.ndarray:
        raise TypeError('Signal data type must be an "ndarray"')
    if signal_in.dtype not in [np.uint16, float]:
        raise TypeError('Signal values must either be "uint16" or "float"')

    # Calculate the number of transients in the signal
    # Characterize the signal
    signal_bounds = (signal_in.min(), signal_in.max())
    unique, counts = np.unique(signal_in, return_counts=True)

    if len(unique) < 10:  # signal is too flat to have a valid peak
        return np.zeros_like(signal_in)

    # Find the peaks
    # i_peaks, _ = find_peaks(signal_in)
    signal_bounds = (signal_in.min(), signal_in.max())
    signal_mean = np.nanmean(signal_in)
    if signal_in.dtype is np.uint16:
        signal_mean = int(np.floor(np.nanmean(signal_in)))
    signal_range = signal_bounds[1] - signal_mean
    prominence = signal_range * 0.8
    distance_min = 10
    i_peaks, properties = find_peaks(signal_in,
                                     height=signal_mean, prominence=prominence,
                                     distance=distance_min)

    if len(i_peaks) == 0:
        raise ArithmeticError('No peaks detected'.format(len(i_peaks), i_peaks))
    if len(i_peaks) > 3:
        print('* {} peaks detected at {} in signal_in'.format(len(i_peaks), i_peaks))
    else:
        raise ValueError('Only {} peak detected at {} in signal_in'.format(len(i_peaks), i_peaks))

    # do not use the first and last peaks
    i_peaks = i_peaks[1:-1]
    # Split up the signal using peaks and estimated cycle length
    est_cycle_array = np.diff(i_peaks).astype(float)
    est_cycle_i = np.nanmean(est_cycle_array)
    est_cycle = est_cycle_i * np.nanmean(np.diff(time_in))
    est_cycle_i = np.floor(est_cycle_i).astype(int)
    cycle_shift = np.floor(est_cycle_i / 2).astype(int)

    signal_time = time_in[0: est_cycle_i]
    signals_trans_peak = []
    i_baselines_full = []
    i_acts_full = []
    signals_trans_act = []

    # roughly isolate all transients centered on their peaks
    # and cropped with a cycle-length-wide window
    # TODO ensembles are too wide due to bad activation times
    # TODO ensembles distorted by early peaks (late activation times?)
    for peak_num, peak in enumerate(i_peaks):
        sig = signal_in[i_peaks[peak_num] - cycle_shift:
                        i_peaks[peak_num] + cycle_shift]
        signals_trans_peak.append(sig)
        # signal = normalize_signal(signal)

        i_baselines = find_tran_baselines(sig)
        i_baselines_full.append((i_peaks[peak_num] - cycle_shift) + i_baselines)
        i_act_signal = find_tran_act(sig)
        i_act_full = (i_peaks[peak_num] - cycle_shift) + i_act_signal
        i_acts_full.append(i_act_full)

    # TODO exclude those with abnormal rise times?

    # With that peak detection, find activation times and align transient
    for act_num, i_act_full in enumerate(i_acts_full):
        if crop == 'center':
            # center : crop transients using the cycle length
            # cropped to center at the alignment points

            # align along activation times, and crop to include ensuing diastolic intervals
            # i_baseline = int(np.median(i_baselines_full[0]))
            i_baseline = int(np.median(i_baselines_full[0]))
            # i_align = i_act_full - (i_acts_full[0] - i_baseline)
            i_start = i_act_full - (i_acts_full[0] - i_baselines_full[0][0])
            i_end = i_act_full + (i_peaks[1] - i_acts_full[0])

            signal_align = signal_in[i_start:i_end]
        elif type(crop) is tuple:
            # stack : crop transients using the cycle length
            # cropped to allow for an ensemble stack with propagating transients

            # Use the earliest end of SNR in the frame

            # stacked to capture the second full transient
            # at the edge of a propagating wave and avoid sliced transients
            # align starting with provided crop times,
            i_align = i_act_full - (i_acts_full[0] - crop[0])
            signal_align = signal_in[i_align:i_align + (crop[1] - crop[0])]

        signal_align = normalize_signal(signal_align)
        # Use correlation to tighten alignment
        shift_max = 0
        if act_num > 0:
            signal_align, shift = align_signals(signals_trans_act[0], signal_align)
            shift_max = np.nanmax([shift_max, abs(shift)])
        signals_trans_act.append(signal_align)

    signals_trans_act = [sig[:-shift_max] for sig in signals_trans_act]
    # use the lowest activation time
    # cycle_shift = min(min(i_acts), cycle_shift)
    # for act_num, act in enumerate(i_acts):
    #     cycle_shift = max(cycle_shift, act)
    #     signals_trans_act.append(signal_in[i_acts[act_num] - cycle_shift:
    #                                        i_acts[act_num] + est_cycle_i - cycle_shift])

    # use the mean of all signals (except the last)
    # TODO try a rms calculation instead of a mean
    signal_out = np.nanmean(signals_trans_act, axis=0)
    signals = signals_trans_act
    i_acts = i_acts_full
    # signal_out = np.nanmean(signals_trans_act, axis=0)
    # signals = signals_trans_act

    return signal_time, signal_out, signals, i_peaks, i_acts, est_cycle


def calc_ensemble_stack(time_in, stack_in):
    """Convert a stack from pixels with multiple transients to those with an averaged signal,
    segmented by activation times. Discards the first and last transients.

        # 1) Confirm the brightest pixel has enough peaks
        # 2) Find pixel(s) with earliest second peak
        # 3) Use the cycle time and end time of that peak's left baseline to align all ensembled signals

        Parameters
        ----------
        time_in : ndarray
            The array of timestamps (ms) corresponding to signal_in, dtyoe : int or float
        stack_in : ndarray
            A 3-D array (T, Y, X) of an optical transient, dtype : uint16 or float

        Returns
        -------
        stack_out : ndarray
             A spatially isolated 3-D array (T, Y, X) of optical data, dtype : float

        Notes
        -----
            Should not be applied to signal data containing at least one transient.
            Pixels with incalculable ensembles are assigned an array of zeros
        """

    print('Ensembling a stack ...')
    map_shape = stack_in.shape[1:]
    i_peak_0_min = stack_in.shape[0]
    yx_peak_1_min = (0, 0)
    i_peak_1_min = stack_in.shape[0]

    # for each pixel ...
    for iy, ix in np.ndindex(map_shape):
        print('\r\tPeak Search of Row:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(
            iy + 1, map_shape[0], ix + 1, map_shape[1]), end='', flush=True)
        # Get first half of signal to save time
        pixel_data = stack_in[:int(stack_in.shape[0]), iy, ix]
        # Characterize the signal
        signal_bounds = (pixel_data.min(), pixel_data.max())
        signal_range = signal_bounds[1] - signal_bounds[0]
        unique, counts = np.unique(pixel_data, return_counts=True)

        if len(unique) < 10:  # signal is too flat to have a valid peak
            continue

        # Find the peaks
        # i_peaks, _ = find_peaks(pixel_data, prominence=signal_range / 4,
        #                         distance=20)
        i_peaks, _ = find_tran_peak(pixel_data, props=True)

        if i_peaks is np.nan:
            continue
        if len(i_peaks) < 4:
            # raise ArithmeticError('No peaks detected'.format(len(i_peaks), i_peaks))
            # np.zeros_like(pixel_data)
            continue
        # if len(i_peaks) > 3:
        #     print('* {} peaks detected at {} in signal_in'.format(len(i_peaks), i_peaks))
        # else:
        #     raise ValueError('Only {} peak detected at {} in signal_in'.format(len(i_peaks), i_peaks))

        # 2) Find pixel(s) with earliest second peak
        # find the first peak and preserve the minimum among all pixels
        if i_peaks[0] < i_peak_0_min:
            i_peak_0_min = i_peaks[0]
            i_peak_1_min = i_peaks[1]
            yx_peak_1_min = (iy, ix)
        # i_peak_1_min = max(i_peaks[1], i_peak_1_min)

    # calculating alignment crop needed to preserve activation propagation
    pixel_data = stack_in[:, yx_peak_1_min[0], yx_peak_1_min[1]]

    i_peaks, _ = find_tran_peak(pixel_data, props=True)

    # Split up the signal using peaks and estimated cycle length
    est_cycle = np.diff(i_peaks).astype(float)
    est_cycle_i = np.nanmean(est_cycle)
    est_cycle_i = np.floor(est_cycle_i).astype(int)
    # est_cycle = est_cycle_i * np.nanmean(np.diff(time_in))
    # cycle_shift = np.floor(est_cycle_i / 2).astype(int)

    peak_1_min_crop = (i_peak_1_min - est_cycle_i, i_peak_1_min + est_cycle_i)
    pixel_data_peak_1_min = pixel_data[peak_1_min_crop[0]: peak_1_min_crop[1]]

    # i_peak_1_min_baselines_l = find_tran_baselines(pixel_data_peak_1_min, peak_side='left')
    # i_peak_1_min_baselines_r = find_tran_baselines(pixel_data_peak_1_min, peak_side='right')

    # ensemble_crop = (i_peak_1_min_baselines_l[-1], i_peak_1_min_baselines_r[1])
    # ensemble_crop = (i_peak_1_min_baselines_l[1] + peak_1_min_crop[0],
    #                  i_peak_1_min_baselines_r[-1] + peak_1_min_crop[0])
    ensemble_crop = peak_1_min_crop
    ensemble_crop_len = ensemble_crop[1] - ensemble_crop[0]

    # 3) Use the cycle time and time of that peak to align all ensembled signals
    # for each pixel ...
    stack_out = np.empty_like(stack_in[:ensemble_crop_len, :, :], dtype=float)

    for iy, ix in np.ndindex(map_shape):
        print('\r\tEnsemble of Row:\t{}\t/ {}\tx\tCol:\t{}\t/ {}'.format(iy + 1, map_shape[0], ix + 1, map_shape[1]),
              end='', flush=True)
        # get signal
        pixel_data = stack_in[:, iy, ix]
        unique, counts = np.unique(pixel_data, return_counts=True)
        if len(unique) < 10:  # signal is too flat to have a valid peak
            signal_ensemble = np.zeros_like(pixel_data[:ensemble_crop_len])
        else:
            # calculate the ensemble of it
            time_ensemble, signal_ensemble, signals, signal_peaks, signal_acts, est_cycle_length \
                = calc_ensemble(time_in, pixel_data, crop=ensemble_crop)

        stack_out[:, iy, ix] = signal_ensemble

    ensemble_yx = yx_peak_1_min
    print('\nDONE Ensembling stack')

    return stack_out, ensemble_crop, ensemble_yx


def calculate_error(ideal, modified):
    """Calculate the amount of error created by signal modulation or filtering,
    defined as (Modified - Ideal) / Ideal X 100%.
    # defined as (Ideal - Modified) / Ideal X 100%.

        Parameters
        ----------
        ideal : ndarray
             An array of ideal data
        modified : ndarray
             An array of modified data

        Returns
        -------
        error : ndarray
             An array of percent errors
        error_mean : float
             The mean value of the percent error array
        error_sd : float
             The standard deviation of the percent error array
        """
    # Check parameters
    if type(ideal) is not np.ndarray:
        raise TypeError('Ideal data type must be an "ndarray"')
    if ideal.dtype not in [int, np.uint16, float]:
        raise TypeError('Ideal values must either be "int", "uint16" or "float"')
    if type(modified) is not np.ndarray:
        raise TypeError('Modified data type must be an "ndarray"')
    if modified.dtype not in [int, np.uint16, float]:
        raise TypeError('Modified values must either be "int", "uint16" or "float"')

    # MIN = 1  # Min to avoid division by 0

    error = ((modified.astype(float) - ideal.astype(float)) / (ideal.astype(float)) * 100)
    error_mean = error.mean()
    error_sd = statistics.stdev(error)

    return error, error_mean, error_sd

def normalize(signal,start,end):
    sig=signal[start:end,:,:]
    aa=np.shape(sig)
    aa=np.array(aa)
    for i in range(aa[1]):
        for j in range(aa[2]): 
            sig[:,i,j]=(sig[:,i,j]-np.amin(sig[:,i,j]))\
                /(np.amax(sig[:,i,j])-np.amin(sig[:,i,j]))
            
    return sig
