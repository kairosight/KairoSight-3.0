from math import pi, floor, ceil, sqrt
import numpy as np
from random import gauss, seed
from scipy import interpolate
from scipy.stats import truncnorm
from scipy.interpolate import UnivariateSpline

# Constants
FL_16BIT_MAX = 2 ** 16 - 1  # Maximum intensity value of a 16-bit pixel: 65535
MIN_TRAN_TOTAL_T = 100  # Minimum transient length (ms)
MIN_APD_20 = 5
MIN_CAD_80 = 50
# Spatial resolution (cm/px)
# resolution = 0.005    # 1 cm / 200 px
# resolution = 0.0149   # pig video resolution
RESOLUTION = 0.01  # 1 cm / 100 px
# Set seed of random number generator
seed(1)
np.random.seed(seed=1)


def model_transients(model_type='Vm', t=100, t0=0, fps=1000, f0=150, famp=100, noise=0,
                     num=1, cl=100, apd=None, cad=None):
    """Create a 2-D array of model 16-bit optical data of either
    murine action potentials (OAP) or murine calcium transients (OCT).

       Parameters
       ----------
       model_type : str
            The type of transient: 'Vm' or 'Ca', default is 'Vm'
       t : int, float
            Length of array in milliseconds (ms), default is 100
       t0 : int or float
            Start time (ms) of first transient, default is 0
       fps : int
            Frame rate (frames per second) of optical data acquisition, default is 1000, min is 200
       f0 : int
            Baseline fluorescence value in counts, default is 150
       famp : int
            Amplitude of the transient in counts, default is 100.
            Can be negative, e.g. cell depolarization with fast voltage dyes
       noise : int
            Magnitude of gaussian noise, as a percentage of f_amp, default is 0
       num : int or str
            Number of transients to generate, default is 1. If 'full', calculate max num to fill array
       cl : int
            Time (ms) between transients aka Cycle Length, default is 100
       apd : dict
            OAP duration times, e.g. APD20, default is {'20': 5}
       cad : dict
            OCT duration times, e.g. CAD40, default is {'80': 15}

       Returns
       -------
       model_time : ndarray
            An array of timestamps (ms) corresponding to the model_data
       model_data : ndarray
            An array of model data, dtype is int
       """
    # Check parameters
    if model_type not in ['Vm', 'Ca']:
        if type(model_type) not in [str]:
            raise TypeError('Model type must be a string, "Vm" or "Ca" ')
        raise ValueError("The model type must either be 'Vm' or 'Ca'")
    if (type(t) is not int) or (type(t0) is not int):
        raise TypeError('All time parameters must be ints')
    if (type(fps) is not int) or (type(f0) is not int) or (type(famp) is not int):
        raise TypeError('All fps and fluorescent parameters must be ints')
    if type(num) not in [int, str]:
        raise TypeError('Number of transients must be an int or "full"')
    if type(cl) not in [int]:
        raise TypeError('Cycle Lenth must be an int')

    if t < MIN_TRAN_TOTAL_T:
        raise ValueError('The time length {} must be longer than {} ms '.format(t, MIN_TRAN_TOTAL_T))
    if t0 >= t:
        raise ValueError('The start time (t0, {}) must be less than the time length (t, {})'.format(t0, t))
    if fps <= 200 or fps > 1000:
        raise ValueError('The fps must be > 200 or <= 1000')

    if famp < 0:
        raise ValueError('The amplitude must >=0')
    if f0 > FL_16BIT_MAX:
        raise ValueError('The baseline fluorescence (f0) must be less than 2^16 - 1 (65535)')
    if abs(famp) > FL_16BIT_MAX:
        raise ValueError('The amplitude (f_amp) must be less than 2^16 - 1 (65535)')
    if f0 + famp > FL_16BIT_MAX:
        raise ValueError('The peak (f_0 + f_amp) must be less than 2^16 - 1 (65535)')
    if f0 + famp + noise > FL_16BIT_MAX:
        raise ValueError('Model data may overflow 16-bit limit: f_0:{}, f_amp:{}, noise:{}'.format(famp, f0, noise))
    if model_type is 'Vm' and (f0 - famp < 0):
        raise ValueError('Effective Vm amplitude is too negative')

    if type(num) not in [str]:
        if num <= 0:
            raise ValueError('The number of transients must be > 0')
        if num * MIN_TRAN_TOTAL_T > t - t0:
            raise ValueError('Too many transients: {}, for total time: {} ms, with start time: {} ms'.format(num, t, t0))
    else:
        if num is not 'full':
            raise ValueError('If not an int, number of transients must be ""full""')

    if cl < 50:
        raise ValueError('The Cycle Length must be > 50 ms')
    if not apd:
        apd = {'20': 5}
    if not cad:
        cad = {'80': MIN_CAD_80}

    # Calculate important constants
    FPMS = fps / 1000
    FRAMES = floor(FPMS * t)
    FRAME_T = 1 / FPMS
    FRAME_T0 = round(t0 / FRAME_T)
    FINAL_T = t - FRAME_T
    if num is 'full':
        num = ceil(FINAL_T / cl)

    # Initialize full model arrays
    model_time = np.linspace(start=0, stop=FINAL_T, num=FRAMES)  # time array
    model_data = np.full(int(FPMS * t), f0, dtype=np.uint16)  # data array, default value is f_0
    if not np.equal(model_time.size, model_data.size):
        raise ArithmeticError('Lengths of time and data arrays not equal!')

    if model_type is 'Vm':
        # With voltage dyes, depolarization transients have a negative deflection and return to baseline
        # Initialize a single OAP array (50 ms) + 50 ms to sync with Ca
        vm_amp = -famp
        # Depolarization phase
        model_dep_period = 5  # XX ms long
        model_dep_frames = floor(model_dep_period / FRAME_T)
        # Generate high-fidelity data
        model_dep_full = np.full(model_dep_period, f0, dtype=np.uint16)
        for i in range(0, model_dep_period):
            model_dep_full[i] = f0 + (vm_amp * np.exp(-(((i - model_dep_period) / 3) ** 2)))  # a simplified Gaussian
        # Under-sample the high-fidelity data
        model_dep = model_dep_full[::floor(model_dep_period / model_dep_frames)][:model_dep_frames]

        # Early repolarization phase (from peak to APD 20, aka 80% of peak)
        model_rep1_period = apd['20']  # XX ms long
        model_rep1_frames = floor(model_rep1_period / FRAME_T)
        apd_ratio = 0.8
        m_rep1 = -(vm_amp - (vm_amp * apd_ratio)) / model_rep1_period  # slope of this phase
        model_rep1 = np.full(model_rep1_frames, f0, dtype=np.uint16)
        for i in range(0, model_rep1_frames):
            model_rep1[i] = ((m_rep1 * i) + vm_amp + f0)  # linear

        # Late repolarization phase
        model_rep2_period = 50 - model_dep_period - model_rep1_period  # remaining OAP time
        model_rep2_frames = floor(model_rep2_period / FRAME_T)
        model_rep2_t = np.linspace(0, 50, model_rep2_frames)
        A, B, C = vm_amp * 0.8, (5 / m_rep1), f0  # exponential decay parameters
        # model_rep2 = A * np.exp(-B * model_rep2_t) + C    # exponential decay, concave down
        tauFall = 10
        model_rep2 = A * np.exp(-model_rep2_t / tauFall) + C  # exponential decay, concave down, using tauFall
        model_rep2 = model_rep2.astype(np.uint16, copy=False)
        # Pad the end with 50 ms of baseline
        model_rep2Pad_frames = floor(50 / FRAME_T)
        model_rep2Pad = np.full(model_rep2Pad_frames, f0, dtype=np.uint16)
        model_rep2 = np.concatenate((model_rep2, model_rep2Pad), axis=None)

    else:
        # With calcium dyes, depolarization transients have a positive deflection and return to baseline
        # Initialize a single OCT array (100 ms)
        # Depolarization phase
        model_dep_period = 10  # XX ms long
        model_dep_frames = floor(model_dep_period / FRAME_T)
        # Generate high-fidelity data
        model_dep_full = np.full(model_dep_period, f0, dtype=np.uint16)
        for i in range(0, model_dep_period):
            model_dep_full[i] = f0 + (famp * np.exp(-(((i - model_dep_period) / 6) ** 2)))  # a simplified Gaussian
        # Under-sample the high-fidelity data
        model_dep = model_dep_full[::floor(model_dep_period / model_dep_frames)][:model_dep_frames]

        # Early repolarization phase (from peak to CAD 80, aka 20% of peak)
        model_rep1_period = floor(cad['80'])  # XX ms long
        model_rep1_frames = floor(model_rep1_period / FRAME_T)
        cad_ratio = 0.2
        m_rep1 = -(famp - (famp * cad_ratio)) / model_rep1_period  # slope of this phase
        model_rep1_full = np.full(model_rep1_period, f0, dtype=np.uint16)
        # Generate high-fidelity data
        for i in range(0, model_rep1_period):
            model_rep1_full[i] = ((m_rep1 * i) + famp + f0)  # linear
        # Under-sample the high-fidelity data
        model_rep1 = model_rep1_full[::floor(model_rep1_period / model_rep1_frames)][:model_rep1_frames]

        # Late repolarization phase
        model_rep2_period = 100 - model_dep_period - model_rep1_period  # remaining OCT time
        model_rep2_frames = floor(model_rep2_period / FRAME_T)
        model_rep2_t = np.linspace(0, 100, model_rep2_frames)
        A, B, C = famp * cad_ratio, (0.8 / m_rep1), f0  # exponential decay parameters
        # model_rep2 = A * np.exp(B * model_rep2_t) + C    # exponential decay, concave up
        tauFall = 30
        model_rep2 = A * np.exp(-model_rep2_t / tauFall) + C  # exponential decay, concave up, using tauFall
        model_rep2 = model_rep2.astype(np.uint16, copy=False)

    # Assemble the transient
    # model_tran = np.concatenate((model_dep, model_rep1, model_rep2), axis=None)
    model_tran = np.concatenate((model_dep, model_rep1, model_rep2), axis=None)

    # Assemble the start time and transient(s) into the full array
    cl_frames = floor(cl / FRAME_T)
    if cl_frames < floor(100 / FRAME_T):
        # Shorten the transient array
        model_tran = model_tran[:cl]
    else:
        # Pad the transient array
        tranPad_frames = floor((cl - 100) / FRAME_T)
        tranPad = np.full(tranPad_frames, f0, dtype=np.uint16)
        model_tran = np.concatenate((model_tran, tranPad), axis=None)

    # Assemble the train of transients
    model_tran_train = np.tile(model_tran, num)
    if model_tran_train.size > model_data.size - FRAME_T0:
        # Shorten train array to fit into final data array
        model_tran_train = model_tran_train[:model_data.size - FRAME_T0]

    model_data[FRAME_T0:FRAME_T0 + model_tran_train.size] = model_tran_train

    # TODO use B-spline to construct the transient
    # B-spline interpolation of model data, 2nd degree spline fitting
    t, c, k = interpolate.splrep(model_time, model_data, k=2)
    N = FRAMES
    # xx = np.linspace(model_time.min(), model_time.max(), N)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    model_data = spline(model_time)

    # Add gaussian noise, mean: 0, standard deviation: noise% of peak, length
    # model_noise = np.random.normal(0, (noise/100) * famp, model_data.size)
    # model_data = model_data + np.round(model_noise)

    # create white noise series
    # white_noise = [gauss(0.0, noise) for i in range(len(model_data))]
    # model_data = model_data + np.round(white_noise)

    # create truncated white noise series
    noise_trunc = 2.5
    if noise > 0:
        model_noise = truncnorm((-noise_trunc * noise - 0.0) / noise,
                                (noise_trunc * noise - 0.0) / noise,
                                    loc=0.0, scale=noise)
        model_data = model_data + np.round(model_noise.rvs(len(model_data)))

    for num, v in enumerate(model_data):
        if abs(v - f0) > (famp * noise_trunc):
            # raise ValueError('All signal values must be >= 0')
            print('* WEIRD value: #{}\t:\t{}'.format(num, v))

    return model_time, model_data.astype(np.uint16)


def model_transients_pig(model_type='Vm', t=150, t0=10, fps=1000, f0=150, famp=100, noise=0,
                         num=1, cl=200, apd=None, cad=None):
    """Create a 2-D array of model 16-bit optical data of either
    pig action potentials (OAP) or pig calcium transients (OCT).

       Parameters
       ----------
       model_type : str
            The type of transient: 'Vm' or 'Ca', default is 'Vm'
       t : int, float
            Length of array in milliseconds (ms), default is 100
       t0 : int or float
            Start time (ms) of first transient, default is 0
       fps : int
            Frame rate (frames per second) of optical data acquisition, default is 1000, min is 200
       f0 : int
            Baseline fluorescence value in counts, default is 150
       famp : int
            Amplitude of the transient in counts, default is 100.
            Can be negative, e.g. cell depolarization with fast voltage dyes
       noise : int
            Magnitude of gaussian noise, as a percentage of f_amp, default is 0
       num : int or str
            Number of transients to generate, default is 1. If 'full', calculate max num to fill array
       cl : int
            Time (ms) between transients aka Cycle Length, default is 200
       apd : dict
            OAP duration times, e.g. APD20, default is {'20': 35}
       cad : dict
            OCT duration times, e.g. CAD40, default is {'40': 15}

       Returns
       -------
       model_time : ndarray
            An array of timestamps (ms) corresponding to the model_data
       model_data : ndarray
            An array of model data, dtype is int
       """
    # Check parameters
    if model_type not in ['Vm', 'Ca']:
        if type(model_type) not in [str]:
            raise TypeError('Model type must be a string, "Vm" or "Ca" ')
        raise ValueError("The model type must either be 'Vm' or 'Ca'")
    if (type(t) is not int) or (type(t0) is not int):
        raise TypeError('All time parameters must be ints')
    if (type(fps) is not int) or (type(f0) is not int) or (type(famp) is not int):
        raise TypeError('All fps and fluorescent parameters must be ints')
    if type(num) not in [int, str]:
        raise TypeError('Number of transients must be an int or "full"')
    if type(cl) not in [int]:
        raise TypeError('Cycle Lenth must be an int')

    if t < MIN_TRAN_TOTAL_T:
        raise ValueError('The time length {} must be longer than {} ms '.format(t, MIN_TRAN_TOTAL_T))
    if t0 >= t:
        raise ValueError('The start time (t0, {}) must be less than the time length (t, {})'.format(t0, t))
    if fps <= 200 or fps > 1000:
        raise ValueError('The fps must be > 200 or <= 1000')

    if famp < 0:
        raise ValueError('The amplitude must >=0')
    if f0 > FL_16BIT_MAX:
        raise ValueError('The baseline fluorescence (f0) must be less than 2^16 - 1 (65535)')
    if abs(famp) > FL_16BIT_MAX:
        raise ValueError('The amplitude (f_amp) must be less than 2^16 - 1 (65535)')
    if f0 + famp > FL_16BIT_MAX:
        raise ValueError('The peak (f_0 + f_amp) must be less than 2^16 - 1 (65535)')
    if f0 + famp + noise > FL_16BIT_MAX:
        raise ValueError('Model data may overflow 16-bit limit: f_0:{}, f_amp:{}, noise:{}'.format(famp, f0, noise))
    if model_type is 'Vm' and (f0 - famp < 0):
        raise ValueError('Effective Vm amplitude is too negative')

    if type(num) not in [str]:
        if num <= 0:
            raise ValueError('The number of transients must be > 0')
        if num * MIN_TRAN_TOTAL_T > t - t0:
            raise ValueError('Too many transients, {}, for total time, {} ms with start time {} ms'.format(num, t, t0))
    else:
        if num is not 'full':
            raise ValueError('If not an int, number of transients must be ""full""')

    if cl < 50:
        raise ValueError('The Cycle Length must be > 50 ms')
    if not apd:
        apd = {'20': 35}
    if not cad:
        cad = {'40': 65}

    # Calculate important constants
    FPMS = fps / 1000
    FRAMES = floor(FPMS * t)
    FRAME_T = 1 / FPMS
    FRAME_T0 = round(t0 / FRAME_T)
    FINAL_T = t - FRAME_T
    max_duration = 120
    if num is 'full':
        num = ceil(FINAL_T / cl)

    # Initialize full model arrays
    model_time = np.linspace(start=0, stop=FINAL_T, num=FRAMES)  # time array
    model_data = np.full(int(FPMS * t), f0, dtype=np.uint16)  # data array, default value is f_0
    if not np.equal(model_time.size, model_data.size):
        raise ArithmeticError('Lengths of time and data arrays not equal!')

    if model_type is 'Vm':
        # With voltage dyes, depolarization transients have a negative deflection and return to baseline
        # Initialize a single OAP array (100 ms)
        vm_duration = 100
        vm_amp = -famp
        # Depolarization phase
        model_dep_period = 10  # XX ms long
        model_dep_frames = floor(model_dep_period / FRAME_T)
        # Generate high-fidelity data
        model_dep_full = np.full(model_dep_period, f0, dtype=np.uint16)
        for i in range(0, model_dep_period):
            model_dep_full[i] = f0 + (vm_amp * np.exp(-(((i - model_dep_period) / 3) ** 2)))  # a simplified Gaussian
        # Under-sample the high-fidelity data
        model_dep = model_dep_full[::floor(model_dep_period / model_dep_frames)][:model_dep_frames]

        # Early repolarization phase (from peak to APD 20, aka 80% of peak)
        model_rep1_period = apd['20']  # XX ms long
        model_rep1_frames = floor(model_rep1_period / FRAME_T)
        apd_ratio = 0.8
        m_rep1 = -(vm_amp - (vm_amp * apd_ratio)) / model_rep1_period  # slope of this phase
        model_rep1 = np.full(model_rep1_frames, f0, dtype=np.uint16)
        for i in range(0, model_rep1_frames):
            model_rep1[i] = ((m_rep1 * i) + vm_amp + f0)  # linear

        # Late repolarization phase
        model_rep2_period = vm_duration - model_dep_period - model_rep1_period  # remaining OAP time
        model_rep2_frames = floor(model_rep2_period / FRAME_T)
        model_rep2_t = np.linspace(0, vm_duration, model_rep2_frames)
        A, B, C = vm_amp * apd_ratio, (5 / m_rep1), f0  # exponential decay parameters
        # model_rep2 = A * np.exp(-B * model_rep2_t) + C    # exponential decay, concave down
        tauFall = 30
        model_rep2 = A * np.exp(-model_rep2_t / tauFall) + C  # exponential decay, concave down, using tauFall

        model_rep2 = model_rep2.astype(np.uint16, copy=False)
        # Pad the end with 50 ms of baseline
        model_rep2Pad_frames = floor(50 / FRAME_T)
        model_rep2Pad = np.full(model_rep2Pad_frames, f0, dtype=np.uint16)
        model_rep2 = np.concatenate((model_rep2, model_rep2Pad), axis=None)

    else:
        # With calcium dyes, depolarization transients have a positive deflection and return to baseline
        # Initialize a single OCT array (120 ms)
        ca_duration = 120
        # Depolarization phase
        model_dep_period = 20  # XX ms long
        model_dep_frames = floor(model_dep_period / FRAME_T)
        # Generate high-fidelity data
        model_dep_full = np.full(model_dep_period, f0, dtype=np.uint16)
        for i in range(0, model_dep_period):
            model_dep_full[i] = f0 + (famp * np.exp(-(((i - model_dep_period) / 6) ** 2)))  # a simplified Gaussian
        # Under-sample the high-fidelity data
        model_dep = model_dep_full[::floor(model_dep_period / model_dep_frames)][:model_dep_frames]

        # Early repolarization phase (from peak to CAD 40, aka 60% of peak)
        model_rep1_period = cad['40']  # XX ms long
        model_rep1_frames = floor(model_rep1_period / FRAME_T)
        cad_ratio = 0.6
        m_rep1 = -(famp - (famp * cad_ratio)) / model_rep1_period  # slope of this phase
        model_rep1_full = np.full(model_rep1_period, f0, dtype=np.uint16)
        # Generate high-fidelity data
        for i in range(0, model_rep1_period):
            model_rep1_full[i] = ((m_rep1 * i) + famp + f0)  # linear, decreasing
        # Under-sample the high-fidelity data
        model_rep1 = model_rep1_full[::floor(model_rep1_period / model_rep1_frames)][:model_rep1_frames]
        # model_rep1 = model_rep1_full[::floor(model_rep1_period/model_rep1_frames)][:model_rep1_frames]

        # Late repolarization phase
        model_rep2_period = ca_duration - model_dep_period - model_rep1_period  # remaining OCT time
        model_rep2_frames = floor(model_rep2_period / FRAME_T)
        model_rep2_t = np.linspace(0, ca_duration, model_rep2_frames)
        A, B, C = famp * cad_ratio, (0.8 / m_rep1), f0  # exponential decay parameters
        # model_rep2 = A * np.exp(B * model_rep2_t) + C    # exponential decay, concave up
        tauFall = 50
        model_rep2 = A * np.exp(-model_rep2_t / tauFall) + C  # exponential decay, concave up, using tauFall
        model_rep2 = model_rep2.astype(np.uint16, copy=False)

    # Assemble the transient
    # model_tran = np.concatenate((model_dep, model_rep1, model_rep2), axis=None)
    model_tran = np.concatenate((model_dep, model_rep1, model_rep2), axis=None)

    # Assemble the start time and transient(s) into the full array
    cl_frames = floor(cl / FRAME_T)
    if cl_frames < floor(100 / FRAME_T):
        # Shorten the transient array
        model_tran = model_tran[:cl]
    else:
        # Pad the transient array
        tranPad_frames = floor((cl - max_duration) / FRAME_T)
        tranPad = np.full(tranPad_frames, f0, dtype=np.uint16)
        model_tran = np.concatenate((model_tran, tranPad), axis=None)

    # Assemble the train of transients
    model_tran_train = np.tile(model_tran, num)
    if model_tran_train.size > model_data.size - FRAME_T0:
        # Shorten train array to fit into final data array
        model_tran_train = model_tran_train[:model_data.size - FRAME_T0]

    model_data[FRAME_T0:FRAME_T0 + model_tran_train.size] = model_tran_train

    # TODO use B-spline to construct the transient
    # B-spline interpolation of model data, 2nd degree spline fitting
    t, c, k = interpolate.splrep(model_time, model_data, k=2)
    N = FRAMES
    # xx = np.linspace(model_time.min(), model_time.max(), N)
    spline = interpolate.BSpline(t, c, k, extrapolate=False)
    model_data = spline(model_time)

    # Add gaussian noise, mean: 0, standard deviation: noise% of peak, length
    model_noise = np.random.normal(0, (noise / 100) * famp, model_data.size)
    # create white noise series
    white_noise = [gauss(0.0, noise) for i in range(len(model_data))]

    model_data = model_data + np.round(white_noise)
    # model_data = model_data + np.round(model_noise)

    return model_time, model_data.astype(np.uint16)


def model_stack(size=(100, 50), **kwargs):
    """Create a stack (3-D array, TYX) of model 16-bit optical data of a
    murine action potential (OAP) or a murine calcium transient (OCT).

       Parameters
       ----------
       size : tuple
            The height and width of the optical data. default is (100, 50)

       Other Parameters
       ----------------
       **kwargs : `.model_transients`. parameter, optional
            All parameters supported by `.model_transients`.

       Returns
       -------
       model_time : ndarray
            An array of timestamps corresponding to model_data
       model_data : ndarray
            A 3-D array (T, Y, X) of model 16-bit data, dtype is int
       """
    # Constants
    MIN_SIZE = (10, 10)  # Minimum stack size (Height, Width)
    # Check parameters
    if type(size) not in [tuple]:
        raise TypeError('Image size must be a tuple, e.g. (20, 20)')
    if (size[0] < MIN_SIZE[0]) or (size[1] < MIN_SIZE[1]):
        raise ValueError('The size (H, W) must be larger than {}'.format(MIN_SIZE))

    # Create a model transient array for each pixel
    pixel_time, pixel_data = model_transients(**kwargs)

    # Initialize full model arrays
    FRAMES = pixel_data.size
    model_time = pixel_time
    model_size = (FRAMES, size[0], size[1])
    model_data = np.empty(model_size, dtype=np.uint16)  # data array, default value is f_0
    for i_frame in range(0, FRAMES):
        # Set every pixel value in that of the model transient
        model_data[i_frame, :, :] = np.full(size, pixel_data[i_frame], dtype=np.uint16)

    return model_time, model_data.astype(np.uint16)


# TODO test SNR map with d_amp
# TODO add variation along propagation (noise, amplitude, duration, Tau) to validate mapping
def model_stack_propagation(size=(100, 50), velocity=20,
                            d_noise=0, d_amp=0, d_dur=0, **kwargs):
    """Create a stack (3-D array, TYX) of model 16-bit optical data of a propagating
    murine action potential (OAP) or a propagating murine calcium transient (OCT).

       Parameters
       ----------
       size : tuple
            The height and width (px) of the optical data. default is (100, 50)
       velocity : int
            Velocity (cm/s) of propagating OAPs/OCTs, default is 20
       d_noise : int
            Units of Noise SD to increase along propagation, default is 0
       d_amp : int
            Units of Amplitude to decrease along propagation, default is 0
       d_dur : int
            Units (ms) of Duration to increase along propagation, default is 0

       Other Parameters
       ----------------
       **kwargs : `.model_transients`. parameter, optional
            All parameters supported by `.model_transients`.

       Returns
       -------
       model_time : ndarray
            An array of timestamps corresponding to model_data
       model_data : ndarray
            A 3-D array (T, Y, X) of model 16-bit data, dtype is int
       """
    # Constants
    # MIN_TOTAL_T = 500   # Minimum stack length (ms)
    MIN_SIZE = (10, 10)  # Minimum stack size (Height, Width)
    MIN_VELOCITY = 10  # Minimum velocity (cm/s)
    MAX_VELOCITY = 50  # Maximum velocity (cm/s)
    DIV_NOISE = 4  # Divisions of noise variation radiating outward from the center
    # Check parameters
    if type(size) not in [tuple]:
        raise TypeError('Image size must be a tuple, e.g. (20, 20)')
    if (size[0] < MIN_SIZE[0]) or (size[1] < MIN_SIZE[1]):
        raise ValueError('The size (H, W) must be larger than {}'.format(MIN_SIZE))
    if velocity < MIN_VELOCITY:
        raise ValueError('The velocity must be larger than {}'.format(MIN_VELOCITY))

    # Convert velocity from cm/s to px/s
    velocity_px = velocity / RESOLUTION
    MIN_VELOCITY_PX = MIN_VELOCITY / RESOLUTION

    # Dimensions of model data (px)
    HEIGHT, WIDTH = size
    # Allocate space for the Activation Map used for propagation
    act_map = np.zeros(shape=(HEIGHT, WIDTH))
    HEIGHT_cm, WIDTH_cm = HEIGHT * RESOLUTION, WIDTH * RESOLUTION

    # Calculate region borders (as distance from the center) for varying a transient parameter
    div_borders = np.linspace(start=int(HEIGHT / 2), stop=HEIGHT / 2 / DIV_NOISE, num=DIV_NOISE)

    # Calculate the absolute minimum time needed for any single propagation, assuming the max velocity
    max_dimension = np.max((HEIGHT_cm, WIDTH_cm))
    MIN_t = MIN_TRAN_TOTAL_T + floor((max_dimension / MAX_VELOCITY) * 1000)  # ms

    if 't' in kwargs:
        if kwargs.get('t') < MIN_t:
            raise ValueError('The total stack time must be longer than {}'.format(MIN_t))
        t_propagation = kwargs.get('t')
    else:  # Use a calculated min total time
        t_propagation = MIN_TRAN_TOTAL_T + floor(HEIGHT_cm / velocity * 1000)  # ms

    if 't0' in kwargs:
        t_propagation = t_propagation + kwargs.get('t0')

    kwargs['t'] = t_propagation

    # Create a model transient array
    pixel_time, pixel_data = model_transients(**kwargs)

    # Initialize full model arrays
    FRAMES = pixel_data.size
    model_time = pixel_time
    model_size = (FRAMES, size[0], size[1])
    model_data = np.empty(model_size, dtype=np.uint16)  # data array, default value is f_0

    # Generate an isotropic activation map, radiating from the center
    origin_x, origin_y = WIDTH / 2, HEIGHT / 2
    # Assign an activation time to each pixel
    for iy, ix in np.ndindex(act_map.shape):
        # Compute the distance from the center (px)
        d_px = sqrt((abs(origin_x - ix) ** 2 + abs((origin_y - iy) ** 2)))
        # Calculate the time associated with that distance from the point of activation
        act_time = d_px / velocity_px
        # Convert time from s to ms
        act_time = act_time * 1000
        prop_delta = floor(act_time)

        # Get provided kwargs
        # t0
        if 't0' in kwargs:
            prop_delta = prop_delta + kwargs.get('t0')
        #  noise
        if 'noise' in kwargs:
            noise_offset = kwargs.get('noise')
        else:
            noise_offset = 0
        if ('cad' in kwargs) or ('apd' in kwargs):
            dur_offset = kwargs.get('cad')
        else:
            if ('model_type' in kwargs) and (kwargs['model_type'] is 'Ca'):
                dur_offset = MIN_CAD_80
            else:
                dur_offset = MIN_APD_20

        kwargs_new = kwargs.copy()
        # if d_noise:
        #     if d_px % (HEIGHT / DIV_NOISE) == 0:  # at the border of a new division
        #         noise_delta = (d_noise * d_px / HEIGHT)
        #     noise_offset = noise_offset + noise_delta   # add to the noise sd
        #     kwargs_new['noise'] = noise_offset
        for idx, d_div in enumerate(div_borders):
            if (d_px < d_div) or (idx == DIV_NOISE):  # if the pixel is closer than this division's border, or too close
                pass
            else:
                if d_noise:
                    noise_offset = noise_offset + (d_noise / (idx + 1))
                    kwargs_new['noise'] = noise_offset
                    break
                if d_dur:
                    dur_offset = dur_offset + (d_dur / (idx + 1))
                    if ('model_type' in kwargs) and (kwargs['model_type'] is 'Ca'):
                        kwargs_new['cad'] = {'80': dur_offset}
                    else:
                        kwargs_new['apd'] = {'20': dur_offset}
                    break

        kwargs_new['t0'] = prop_delta
        kwargs_new['t'] = t_propagation

        # Create a model transient array for each pixel
        pixel_time, pixel_data = model_transients(**kwargs_new)
        # Set every pixel's values to those of the offset model transient
        model_data[:, iy, ix] = pixel_data

    return model_time, model_data.astype(np.uint16)


def model_stack_heart(size=(100, 100), velocity=20, d_noise=0, **kwargs):
    """Create a stack (3-D array, TYX) of model 16-bit optical data of a propagating
        murine action potential (OAP) or a propagating murine calcium transient (OCT),
        with a "dark" background around the pixels of interest.

           Parameters
           ----------
           # size : tuple
           #      The height and width (px) of the optical data. default is (100, 50)
           # d_noise : int
           #      Units of Noise SD to increase along propagation, default is 0

           Other Parameters
           ----------------
           **kwargs : `.model_transients`. parameter, optional
                All parameters supported by `.model_transients`.

           Returns
           -------
           model_time : ndarray
                An array of timestamps corresponding to model_data
           model_data : ndarray
                A 3-D array (T, Y, X) of model 16-bit data, dtype is int
           """

    # Create a propagating model stack
    model_time, model_data = model_stack_propagation(size, velocity, d_noise, **kwargs)

    # Create a circular mask from the first frame
    nrows, ncols = model_data[0].shape
    row, col = np.ogrid[:nrows, :ncols]
    cnt_row, cnt_col = nrows / 2, ncols / 2
    outer_disk_mask = ((row - cnt_row) ** 2 + (col - cnt_col) ** 2 > (nrows / 2) ** 2)

    # Apply the mask to each frame
    print('* Masking model heart stack ...')
    for idx, frame in enumerate(model_data):
        print('\r\tFrame:\t{}\t/ {}'.format(idx + 1, model_data.shape[0]), end='', flush=True)
        model_data[idx][outer_disk_mask] = 0
    print('\n* DONE Masking model heart stack')

    return model_time, model_data


# Code for example tests
def circle_area(r):
    if r < 0:
        raise ValueError('The radius cannot be negative')

    if type(r) not in [int, float]:
        raise TypeError('The radius must be a non-negative real number')

    return pi * (r ** 2)
