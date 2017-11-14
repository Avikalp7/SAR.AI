"""Multilevel Local Pattern Histogram for SAR Image Classification.

Based on published research by:
Dai, Dengxin, Wen Yang, and Hong Sun. "Multilevel local pattern histogram for
SAR image classification." IEEE Geoscience and Remote Sensing Letters 8.2
(2011): 225-229.
"""

from __future__ import division
from __future__ import print_function

import math
import numpy as np
from scipy.ndimage import measurements
import scipy.misc

_DEFAULT_THRESHOLD = [3, 8, 16, 32, 64]
_FOUR_CONNECTIVITY = np.array([[0, 1, 0],
                               [1, 1, 1],
                               [0, 1, 0]])


def rgb2gray(rgb):
    """Converts RGB images to gray scale.

    Converts the true-color image RGB to the gray-scale intensity image I. The
    rgb2gray function converts RGB images to gray-scale by eliminating the hue
    and saturation information while retaining the luminance.

    The coefficients used to calculate gray-scale values in rgb2gray are
    identical to those used to calculate luminance (E'y) in Rec.ITU-R BT.601-7
    after rounding to 3 decimal places.

    Rec.ITU-R BT.601-7 calculates E'y using the following formula:
        0.299 * R + 0.587 * G + 0.114 * B

    :param rgb: Matrix with image pixels RGB value.
    :return: Matrix with converted gray scale image value.
    """

    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def read_img(image_name='sar1.tif'):
    """Read image input matrix from image file.

    :param image_name: String with absolute path of image to be read.
    :return: image_input_matrix: Numpy matrix with image pixel values.
    """

    image_input_matrix = scipy.misc.imread(image_name, mode='L')
    num_pixels = 200
    image_input_matrix = image_input_matrix[
                            119:120 + num_pixels,
                            119:120 + num_pixels
                         ]
    return image_input_matrix


def mlph(data, h=7, t=_DEFAULT_THRESHOLD):
    """Computes Local Pattern Histogram for data matrix.

    Calculates LPH for each pixel in the data matrix, based on the specified
    h x h window size of pixel neighbors, at each threshold level in t, combined
    based on sub-histogram binning and concatenation to get multilevel local
    pattern histogram. Fringes of width half of window size are ignored from
    all 4 sides of the image to use a centered data matrix.

    :param data: Input image patch/pixel value matrix of [H, W] size.
    :param h: Integer with window size. 7 is used as default value.
    :param t: List of integers with threshold for intensity contrast. Default
        thresholds are given by [3, 8, 16, 32, 64].

    :raises ValueError: Invalid size of image data matrix (H < h or W < h).
    :raises ValueError: Even window size is passed as argument for h.
    :raises ValueError: Intensity contrast threshold list t is empty.

    :return: texture: Numpy matrix of the shape (len(t) * 3 * b, 1) with MLPH
        texture of input image data matrix.
    """

    print("\nStarting MLPH...")
    data = np.array(data)

    try:
        H, W, Z = data.shape
    except ValueError:
        H, W = data.shape
        Z = 0

    if Z == 3:
        data = rgb2gray(data)

    data.astype(float)

    if H < h or W < h:
        raise ValueError('Invalid size of image data matrix. Image size (H, W) '
                         '= ({H}, {W}), window size h = {h}.'.
                         format(H=H, W=W, h=h))

    if h % 2 == 0:
        raise ValueError('Window size h must be odd. Given h = {h}'.format(h=h))

    if len(t) < 1:
        raise ValueError('Intensity contrast threshold list must have at least'
                         ' one element. t = {t}'.format(t=t))

    # Data pre-processing and preparation
    data_centered = data[
                        int((h - 1) / 2): H - int((h - 1) / 2),
                        int((h - 1) / 2): W - int((h - 1) / 2),
                    ]

    try:
        H_rep, W_rep, _ = data_centered.shape
    except ValueError:
        H_rep, W_rep = data_centered.shape

    rep_data = np.zeros((H_rep, W_rep, h ** 2))

    for i in range(1, h + 1):
        for j in range(1, h + 1):
            rep_data[:, :, (i - 1) * h + j - 1] = (
                data[j - 1: j - 1 + W_rep, i - 1: i - 1 + W_rep])
            rep_data[:, :, (i - 1) * h + j - 1] = (
                rep_data[:, :, (i - 1) * h + j - 1] - data_centered)

    b = int(math.floor(math.log(h ** 2 + 1, 2) + 1))
    pixel_textures = np.zeros((W_rep, H_rep, len(t) * 3 * b))

    # Handle each intensity contrast threshold level
    for n, T in enumerate(t, 1):
        positive_matrix = (rep_data > T).astype(int)
        negative_matrix = (rep_data < -T).astype(int)
        equal_matrix = ((rep_data <= T) & (rep_data >= -T)).astype(int)

        positive_matrix = positive_matrix.reshape(H_rep, W_rep, h, h)
        negative_matrix = negative_matrix.reshape(H_rep, W_rep, h, h)
        equal_matrix = equal_matrix.reshape(H_rep, W_rep, h, h)

        for i in range(1, W_rep + 1):
            for j in range(1, H_rep + 1):

                texture = np.zeros((len(t) * 3 * b, 1))

                # Update texture from positive part of the matrix
                BW = np.squeeze(positive_matrix[j - 1, i - 1, :, :])
                L, num_segments = measurements.label(
                    BW, structure=_FOUR_CONNECTIVITY)
                segments = np.unique(L)

                if num_segments >= 1:
                    for k in range(1, num_segments + 1):
                        segment_length = np.where(L == segments[k])[0].size
                        texture[(n - 1) * 3 * b + int(math.floor(
                            math.log(segment_length, 2) + 1)) - 1] += 1

                # Update texture from negative part of the matrix
                BW = np.squeeze(negative_matrix[j - 1, i - 1, :, :])
                L, num_segments = measurements.label(
                    BW, structure=_FOUR_CONNECTIVITY)
                segments = np.unique(L)

                if num_segments >= 1:
                    for k in range(1, num_segments + 1):
                        segment_length = np.where(L == segments[k])[0].size
                        texture[(n - 1) * 3 * b + b + int(math.floor(
                            math.log(segment_length, 2) + 1)) - 1] += 1

                # Update texture from equal part of the matrix
                BW = np.squeeze(equal_matrix[j - 1, i - 1, :, :])
                L, num_segments = measurements.label(
                    BW, structure=_FOUR_CONNECTIVITY)
                segments = np.unique(L)

                if num_segments >= 1:
                    # for k in range(1, num_segments + 1):
                    for k in range(0, num_segments):
                        segment_length = np.where(L == segments[k])[0].size
                        texture[(n - 1) * 3 * b + 2 * b + int(math.floor(
                            math.log(segment_length, 2) + 1)) - 1] += 1
                texture = texture.reshape(-1, )
                pixel_textures[i - 1, j - 1, :] = texture

        print("Completed %d of %d iterations for current image." % (n, len(t)))

    print("\nCompleted MLPH computation for current image.")
    return pixel_textures, H_rep, W_rep
