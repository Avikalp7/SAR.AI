from __future__ import division
from __future__ import print_function

import scipy.misc
from skimage.viewer import ImageViewer

import os.path


def read_image(image_name):
    image_input_matrix = scipy.misc.imread(image_name, mode='L')
    return image_input_matrix


def find_filename_iteratively(start, end, dir_path):
    i = 1
    while os.path.exists(dir_path + start + str(i) + end):
        i += 1
        continue
    return dir_path + start + str(i) + end


def show_image(image):
    viewer = ImageViewer(image)
    viewer.show()
    return viewer
