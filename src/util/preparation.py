import os
import time
# from memory_profiler import profile
from math import floor
import numpy as np
from pathlib import Path, PurePath
from imageio import volread, volwrite, get_reader
from skimage.util import img_as_uint, img_as_float
from skimage.transform import rescale
from skimage.filters import sobel, threshold_otsu, threshold_mean
from skimage.segmentation import random_walker
# TODO Try felzenszwalb edge filter (https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_segmentations.html#sphx-glr-auto-examples-segmentation-plot-segmentations-py)
# TODO Try Canny edge filter (https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_metrics.html#sphx-glr-auto-examples-segmentation-plot-metrics-py)
from skimage.exposure import rescale_intensity
from skimage.measure import label, regionprops
from skimage.morphology import (opening, flood_fill, square, disk, dilation,
                                erosion, closing, diamond)
import cv2

# Constants
FL_16BIT_MAX = 2 ** 16 - 1  # Maximum intensity value of a 16-bit pixel: 65535
MASK_TYPES = ['Otsu_global', 'Mean', 'Random_walk', 'best_ever', 'Bkgd_thresh']
MASK_STRICT_MAX = 9

# TODO move "reduce_stack" from test_Map setUps to a preparation as a new
# function


def open_signal(source, fps=500):
    """Open an array of optical data from a text file (.csv)
    as a calculated time array and an array of 16-bit arbitrary fluorescent
    data

        Parameters
        ----------
        source : str
            The full path to the file
        fps : int
            The framerate of the recorded captured

        Returns
        -------
        signal_time : ndarray
            An array of timestamps (ms) corresponding to the model_data
        signal_data : ndarray
            An array of normalized fluorescence data
    """
    # Check parameter types
    if type(source) not in [str]:
        raise TypeError('Required "source" ' + source +
                        ' parameter must be a string')
    # Check parameter validity
    # Make sure the source is an existing file
    if not os.path.isfile(source):
        raise FileNotFoundError('Required "source" ' + source +
                                ' is not a file or does not exist.')

    # Load the text file
    signal_text = np.genfromtxt(source, delimiter=',')

    # Calculate important constants
    # Generate array of timestamps
    fpms = fps / 1000

    if len(signal_text.shape) > 1:
        # Multiple columns
        # rows of the first column (skip X,Y header row)
        data_x = signal_text[1:, 0]
        # rows of the first column (skip X,Y header row)
        data_y_counts = signal_text[1:, 1].astype(np.uint16)
        n_frames = len(data_x)
        t_final = floor(n_frames / fpms)
    else:
        # Single column, data only
        data_y_counts = signal_text[0]
        n_frames = len(data_y_counts)
        t_final = floor(n_frames / fpms)

    signal_data = data_y_counts
    # Generate array of timestamps
    signal_time = np.linspace(start=0, stop=t_final, num=n_frames)

    return signal_time, signal_data


def open_stack(source, meta=None):
    """Open a stack of images (.tif, .tiff, .pcoraw) from a file.

       Parameters
       ----------
       source : str
            The full path to the file
       meta : str, optional
            The full path to a file containing metadata

       Returns
       -------
       stack : ndarray
            A 3-D array (T, Y, X) of optical data, 16-bit
       meta : dict
            A dict of metadata

        Notes
        -----
            Files with .pcoraw extension are converted and saved as .tif.
            Expecting volume dimension order XYCZT
       """
    # Check parameter types
    if type(source) not in [str]:
        raise TypeError(
            'Required "source" ' + source + ' parameter must be a string')
    if meta and (type(meta) not in [str]):
        raise TypeError(
            'Optional "meta" ' + meta + ' parameter must be a string')

    # Check validity
    # Make sure the directory, source file, and optional meta file exists
    if not os.path.isdir(os.path.split(source)[0]):
        raise FileNotFoundError(
            'Required directory ' + os.path.split(source)[0]
            + ' is not a directory or does not exist.')
    if not os.path.isfile(source):
        raise FileNotFoundError(
            'Required "source" ' + source
            + ' is not a file or does not exist.')
    if meta and not os.path.isfile(meta):
        raise FileNotFoundError(
            'Optional "meta" ' + meta + ' is not a file or does not exist.')
    # If a .pcoraw file, convert to .tiff
    f_purepath = PurePath(source)
    f_extension = f_purepath.suffix
    if f_extension == '.pcoraw':
        p = Path(source)
        p.rename(p.with_suffix('.tif'))
        source = os.path.splitext(source)[0] + '.tif'
        print('* .pcoraw covnerted to a .tif')
    # Open the metadata, if provided
    stack_meta = get_reader(source, mode='v').get_meta_data()
    # Open the file
    # file_source = open(source, 'rb')
    # tags = exifread.process_file(file)  # Read EXIF data
    stack = volread(source)  # Read image data, closes the file after read
    stack = img_as_uint(stack)  # Read image data, closes the file after read
    if meta:
        file_meta = open(meta)
        meta = file_meta.read()
        file_meta.close()
    else:
        meta = stack_meta
    return stack, meta


# def crop_frame(frame_in, d_x, d_y):
#     frame_out = frame_in.copy()
#
#     if (d_x > 0) and (d_y > 0):
#         frame_out = frame_out[0:-d_y, 0:-d_x]
#     else:
#         if d_x < 0:
#             frame_out = frame_out[:, -d_x:]
#         if d_y < 0:
#             frame_out = frame_out[-d_y:, :]
#
#     return frame_out


def crop_stack(stack_in, d_x=False, d_y=False):
    """Crop a stack (3-D array, TYX) of optical data,
    by default removes from right and bottom.

       Parameters
       ----------
       stack_in : ndarray
            A 3-D array (T, Y, X) of optical data, dtype : uint16 or float
       d_x : int
            Amount of pixels to remove from the input's width.
            < 0 to crop from the left, > 0 to crop from the right
       d_y : int
            Amount of pixels to remove from the input's height.
            < 0 to crop from the top, > 0 to crop from the bottom

       Returns
       -------
       stack_out : ndarray
            A cropped 3-D array (T, Y, X) of optical data, dtype : stack_in.dtype
       """
    # Check parameters
    if type(stack_in) is not np.ndarray:
        raise TypeError('Stack type must be an "ndarray"')
    if len(stack_in.shape) != 3:
        raise TypeError('Stack must be a 3-D ndarray (T, Y, X)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')
    # if type(d_x) is not int:
    #     raise TypeError('X pixels to crop must be an "int"')
    # if type(d_y) is not int:
    #     raise TypeError('Y pixels to crop must be an "int"')

    stack_out = []
    # if either X or Y crop is unused, set to 0
    if d_x is False:
        d_x = 0
    if d_y is False:
        d_y = 0

    if (d_x > 0) and (d_y > 0):
        stack_out = stack_in[:, 0:-d_y, 0:-d_x]
    else:
        if d_x < 0:
            stack_out = stack_in[:, :, -d_x:]
            stack_in = stack_out
        elif d_x > 0:
            stack_out = stack_in[:, :, 0:-d_x:]
            stack_in = stack_out

        if d_y < 0:
            stack_out = stack_in[:, -d_y:, :]
        elif d_y > 0:
            stack_out = stack_in[:, 0:-d_y:, :]

    return stack_out


def reduce_stack(stack_in, reduction=1):
    """Rescale the X,Y dimensions of a stack (3-D array, TYX) of optical data,
    using linear interpolation and gaussian anti-aliasing to effectively bin pixels together.

       Parameters
       ----------
       stack_in : ndarray
            A 3-D array (T, Y, X) of optical data, dtype : uint16 or float
       reduction : int, float
            Factor by which to reduce both dimensions, typically in the range 2-10

       Returns
       -------
       stack_out : ndarray
            A reduced 3-D array (T, Y, X) of optical data, dtype : stack_in.dtype
       """
    reduction_factor = 1 / reduction
    test_frame_reduced = rescale(stack_in[0], reduction_factor, multichannel=False)
    stack_reduced_shape = (stack_in.shape[0], test_frame_reduced.shape[0], test_frame_reduced.shape[1])
    stack_out = np.empty(stack_reduced_shape, dtype=stack_in.dtype)  # empty stack
    print('Reducing stack from W {} X H {} ... to size W {} X H {} ...'
          .format(stack_in.shape[2], stack_in.shape[1], test_frame_reduced.shape[1], test_frame_reduced.shape[0]))
    for idx, frame in enumerate(stack_in):
        # print('\r\tFrame:\t{}\t/ {}'.format(idx + 1, stack_in.shape[0]), end='', flush=True)
        #     f_filtered = filter_spatial(frame, kernel=self.kernel)
        frame_reduced = img_as_uint(rescale(frame, reduction_factor, anti_aliasing=True, multichannel=False))
        stack_out[idx, :, :] = frame_reduced

    return stack_out


def mask_generate(frame_in, mask_type='Otsu_global', strict=(3, 5)):
    """Generate a mask for a frame 2-D array (Y, X) of grayscale optical data
    using binary threshold (histogram-based or local) or segmentation algorithms.

       Parameters
       ----------
       frame_in : ndarray
            A 2-D array (Y, X) of optical data, dtype : uint16 or float
       mask_type : str  # TODO add masking via SNR
            The type of masking thresholding algorithm to use, default : Otsu_global
       strict : tuple, optional
            How strict to be with adjustable masking (dark, light), default : (3, 5)
            Ranges from 1 to 9, higher means brighter cutoff.
            Dark must be less than light: dark < light

       Returns
       -------
       frame_out : ndarray
            A 2-D array (Y, X) of masked optical data,  dtype : frame_in.dtype
       mask : ndarray
            A binary 2-D array generated from the threshold algorithm, dtype : np.bool_
       markers : ndarray
            A 2-D array of markers generated during a masking algorithm, dtype : frame_in.dtype or float
       """
    # Check parameters
    if type(frame_in) is not np.ndarray:
        raise TypeError('Frame type must be an "ndarray"')
    '''if len(frame_in.shape) != 2:
        raise TypeError('Frame must be a 2-D ndarray (Y, X)')'''
    if frame_in.dtype not in [np.uint16, float]:
        raise TypeError('Frame values must either be "np.uint16" or "float"')
        raise TypeError('Filter type must be a "str"')
    if type(strict) is not tuple:
        raise TypeError('Strictness type must be an "tuple"')
    if len(strict) != 2:
        raise TypeError('Strictness length must be 2')
    if mask_type != 'Bkgd_thresh' and strict[0] > strict[1]:
        raise TypeError(
            'Strictness for Dark cutoff must be greater than Light cutoff')
    if mask_type != 'Bkgd_thresh' and strict[0] < 1:
        raise TypeError('Strictness for Dark cutoff must be greater than 0')
    if mask_type == 'Bkgd_thresh' and strict[0] > 1:
        raise TypeError(
            'Strictness for Percentage must be less than 1.')
    if mask_type == 'Bkgd_thresh' and strict[0] < 0:
        raise TypeError('Strictness for Percentage must be greater than 0')
    if mask_type not in MASK_TYPES:
        raise ValueError(
            'Filter type must be one of the following: {}'.format(MASK_TYPES))

    frame_out = frame_in.copy()
    mask = frame_in.copy()
    markers = np.zeros(frame_in.shape)

    frame_in_gradient = sobel(frame_in)

    if mask_type == 'Otsu_global':
        # Good for ___, but ___
        global_otsu = threshold_otsu(frame_in)
        print(f'Global Otsu: {global_otsu}')
        binary_global = frame_in >= global_otsu
        mask = binary_global
        frame_out[mask] = 0

    elif mask_type == 'Mean':
        # Good for ___, but __
        thresh = threshold_mean(frame_in)
        binary_global = frame_in >= thresh
        mask = binary_global
        frame_out[mask] = 0

    elif mask_type == 'Random_walk':
        # https://scikit-image.org/docs/0.13.x/auto_examples/segmentation/plot_random_walker_segmentation.html
        # The range of the binary image spans over (-1, 1)
        # We choose extreme tails of the histogram as markers, and use diffusion to fill in the rest.
        frame_in_float = img_as_float(frame_in)
        frame_in_rescale = rescale_intensity(frame_in_float,
                                             in_range=(frame_in_float.min(), frame_in_float.max()),
                                             out_range=(-1, 1))
        markers = np.zeros(frame_in_rescale.shape)
        otsu = threshold_otsu(frame_in_rescale, nbins=256 * 2)

        # Calculate thresholds between -1 and otsu: darkest to lightest
        num_otsus = MASK_STRICT_MAX + 1       # number of sections between -1 and otsu
        otsus = np.linspace(-1, otsu, num=num_otsus)

        print('* Masking otsu choices: {}'.format([round(ots, 3) for ots in otsus]))
        markers_dark_cutoff = otsus[strict[0]]      # darkest section (< first otsu section)
        markers_light_cutoff = otsus[strict[1]]   # lightest section (> #strictness otsu section)

        print('\t* Marking Random Walk with Otsu values: {} & {}'
              .format(round(markers_dark_cutoff, 3), round(markers_light_cutoff, 3)))

        markers[frame_in_rescale < markers_dark_cutoff] = 1
        markers[frame_in_rescale > markers_light_cutoff] = 2

        # Run random walker algorithm
        binary_random_walk = random_walker(frame_in_rescale, markers, mode='bf')
        # Keep the largest bright region
        labeled_mask = label(binary_random_walk)
        largest_mask = np.empty_like(labeled_mask, dtype=np.bool_)
        largest_region_area = 0
        for idx, region_prop in enumerate(regionprops(labeled_mask)):
            # Use the biggest bright region

            # for prop in region_prop:
            #     print(prop, region_prop[prop])
            # use the second-largest region
            # print('* Region #{}\t:\tint: _\tarea: {}'
            #       .format(idx + 1, region_prop.area))
            if region_prop.area < 2:
                pass
            if region_prop.area > largest_region_area and region_prop.label > 1:
                largest_region_area = region_prop.area
                largest_mask[labeled_mask == region_prop.label] = False
                largest_mask[labeled_mask != region_prop.label] = True
                print('\t* Using #{} area: {}'
                      .format(idx+1, region_prop.area))

        frame_out[largest_mask] = 0
        mask = largest_mask
    elif mask_type == 'Bkgd_thresh':
        # Grab the min and max for all signals
        data_max = frame_in.max(axis=0)
        data_min = frame_in.min(axis=0)
        # Calculate the amplitude of signals using min and max values
        data_range = data_max-data_min
        # Rearrange the data to enable sorting
        to_sort = np.reshape(
            data_range, [1, data_range.shape[0]*data_range.shape[1]])
        to_sort = np.sort(to_sort, axis=1, )
        to_sort = np.flip(to_sort)
        # Grab the amplitude value at the % threshold (i.e., strict[0])
        range_thresh = to_sort[0, np.int(to_sort.shape[1]*strict[0])]
        # Use the % threshold amplitude value to generate a mask inclusive of
        # all pixels above the top % of values
        thresh_mask = data_range > range_thresh
        # Create kernels for morphological manipulation of the mask
        selem_duo = diamond(strict[1])
        selem_solo = diamond(strict[1]*3)
        # Close the mask to fill holes
        thresh_mask = closing(thresh_mask, selem_duo)
        # Open the mask to prune away edges
        thresh_mask = opening(thresh_mask, selem_duo)
        # Dilate to fill the space
        thresh_mask = dilation(thresh_mask, selem_solo)
        # Use connected component analysis to isolate heart
        labels = label(thresh_mask)
        labels_vect = np.reshape(labels, [1, labels.shape[0]*labels.shape[1]])
        labels_count = np.zeros([1, labels_vect.max()])
        for n in np.arange(0, labels_vect.max()):
            labels_count[0, n] = np.sum(labels == n+1)
            label_ind = np.argmax(labels_count)
            thresh_mask = labels == label_ind+1
        # Apply mask to data
        frame_out[0][thresh_mask] = 0
        # Assign ouput variable
        mask = thresh_mask
    else:
        raise NotImplementedError(
            'Mask type "{}" not implemented'.format(mask_type))

    return frame_out, mask, markers


def mask_apply(stack_in, mask, invert):
    """Apply a binary mask to segment a stack (3-D array, TYX) of grayscale optical data.

       Parameters
       ----------
       stack_in : ndarray
            A 3-D array (T, Y, X) of optical data, dtype : uint16 or float
       mask : ndarray
            A binary 2-D array (Y, X) to mask optical data, dtype : np.bool_
       invert : int
            The currentIndex from the voltage/calcium drop down

       Returns
       -------
       stack_out : ndarray
            A masked 3-D array (T, Y, X) of optical data, dtype : stack_in.dtype
            Masked values are FL_16BIT_MAX (aka 65535)
       """
    # Check parameters
    if type(stack_in) is not np.ndarray:
        raise TypeError('Stack type must be an "ndarray"')
    if len(stack_in.shape) != 3:
        raise TypeError('Stack must be a 3-D ndarray (T, Y, X)')
    if stack_in.dtype not in [np.uint16, float]:
        raise TypeError('Stack values must either be "np.uint16" or "float"')

    if type(mask) is not np.ndarray:
        raise TypeError('Mask type must be an "ndarray"')
    if mask.dtype not in [np.int64, bool]:
        raise TypeError('Stack values must either be "np.bool_"')
    if len(mask.shape) != 2:
        raise TypeError('Mask must be a 2-D ndarray (Y, X)')

    frame_0 = stack_in[0]

    # if (mask.shape[0] is not frame_0.shape[0]) or (mask.shape[1] is not frame_0.shape[1]):
    if mask.shape != frame_0.shape:
        raise ValueError('Mask shape must be the same as the stack frames:'
                         '\nMask:\t{}\nFrame:\t{}'.format(
                             mask.shape, frame_0.shape))

    stack_out = np.empty_like(stack_in)

    for i_frame, frame in enumerate(stack_in):
        frame_out = frame.copy()
        if invert == 0:
            frame_out[mask] = 0
        else:
            frame_out[~mask] = 0
        stack_out[i_frame] = frame_out

    return stack_out


def get_gradient(im):
    # Calculate the x and y gradients using a Sobel operator
    grad_x = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize=5)

    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    return grad


def align_stacks(stack1, stack2):
    """Aligns two stacks of images using the gradient representation of the image
    and a similarity measure called Enhanced Correlation Coefficient (ECC).
    TODO try Feature-Based approach https://www.learnopencv.com/image-alignment-feature-based-using-opencv-c-python/, https://github.com/spmallick/learnopencv/blob/c8e3ae2d2b0423f5c6d21c6189ee8ff3192c0555/ImageAlignment-FeatureBased/align.py

        Parameters
        ----------
        stack1 : ndarray, dtype : uint16
            Image stack with shape (t, y, x)
        stack2 : ndarray, dtype : uint16
            Image stack with shape (t, y, x), will be aligned to stack1

        Returns
        -------
        stack2_aligned : ndarray
            Aligned version of stack2

        Notes
        -----
        # Assumes differences are translational with no rotation
        # Based on examples by Satya Mallick (https://www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/)
    """
    # Read uint16 grayscale images from the image stacks
    im1 = stack1[0, ...]
    im2 = stack2[0, ...]

    # Find the width and height of the image
    width, height = im1.shape[1], im1.shape[0]
    print('im1 min, max: ', np.nanmin(im1), ' , ', np.nanmax(im1))
    print('im2 min, max: ', np.nanmin(im2), ' , ', np.nanmax(im2))
    # Find the number of frames in the stacks (should be identical)
    frames = stack1.shape[0]

    # Allocate space for aligned image and stack
    im2_aligned = np.zeros_like(im2)
    stack2_aligned = np.zeros_like(stack2)

    # Define motion model
    warp_mode = cv2.MOTION_TRANSLATION
    # Define 2x3 matrices and initialize the matrix to identity
    warp_matrix = np.eye(2, 3, dtype=np.float32)
    # Specify the number of iterations
    number_of_iterations = 5000
    # Specify the threshold of the increment in the correlation coefficient between two iterations
    termination_eps = 1e-10
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                number_of_iterations, termination_eps)

    start = time.time()
    # Warp the second stack image to the first
    # Run the ECC algorithm, the results are stored in warp_matrix
    (cc, warp_matrix) = cv2.findTransformECC(get_gradient(im1), get_gradient(im2),
                                             warp_matrix, warp_mode, criteria)
    # Use Affine warp when the transformation is not a Homography
    im2_aligned = cv2.warpAffine(im2, warp_matrix, (width, height),
                                 flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    print('im2_aligned min, max: ', np.nanmin(im2_aligned), ' , ', np.nanmax(im2_aligned))
    # Convert aligned image back to uint16
    im2_aligned = np.uint16(im2_aligned)
    print('im2_aligned min, max: ', np.nanmin(im2_aligned), ' , ', np.nanmax(im2_aligned))

    # Align and save every stack2 frame using the same process
    for i in range(frames):
        # Find the old frame
        stack2_frame = stack2[i, :, :]
        stack2_frame_aligned = cv2.warpAffine(stack2_frame, warp_matrix, (width, height),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        # Save the aligned frame in the new stack
        stack2_aligned[i, :, :] = stack2_frame_aligned

    print('stack2_aligned min, max: ', np.nanmin(stack2_aligned), ' , ', np.nanmax(stack2_aligned))
    end = time.time()
    print('Alignment time (s): ', end - start)

    return stack2_aligned
