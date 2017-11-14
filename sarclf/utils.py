"""File with helper and utility functions to read and display images, and
iteratively find file names in given directories.
"""

from __future__ import division
from __future__ import print_function

import scipy.misc
from skimage import viewer

import os.path


def read_image(image_name):
    """Reads image and converts into input image matrix.

    :param image_name: String with name of image to be read.
    :return: Input image matrix.
    """

    image_input_matrix = scipy.misc.imread(image_name, mode='L')
    return image_input_matrix


def find_filename_iteratively(start, file_path_suffix, dir_path):
    """Iterates through files in the base path directory to find files.

    :param start: String with starting folder prefix.
    :param file_path_suffix: String with file end/suffix.
    :param dir_path: String with file directory path.
    :return: String with found file path.
    """

    i = 1
    while os.path.exists(dir_path + start + str(i) + file_path_suffix):
        i += 1
        continue

    return dir_path + start + str(i) + file_path_suffix


def show_image(image):
    """Displays image on screen with matplotlib-based canvas.

    :param image: Image data
    :return: image viewer object
    """

    display_image = viewer.ImageViewer(image)
    display_image.show()
    return display_image
