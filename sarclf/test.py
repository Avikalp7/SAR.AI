from __future__ import division
from __future__ import print_function

from sarclf import mlph
from sarclf import mlph_modified
from sarclf import utils

import numpy as np
import skimage.color
import skimage.io
from sklearn.metrics import classification_report

import warnings
import os.path


def test_pixel_classification(X_test, y_test, clf):
    """Computes classification accuracy of pixel classification.

    :param X_test: Test set features matrix
    :param y_test: Test set target labels matrix
    :param clf: Trained classifier object
    :return: Float with accuracy score in percent
    """

    Y_true, Y_pred = y_test, clf.predict(X_test)

    print("Report: ")
    print("\n{}".format(classification_report(Y_true, Y_pred, digits=4)))

    accuracy = clf.score(X_test, y_test)
    print("Total accuracy on test set = %0.2f" % (accuracy * 100))

    return accuracy * 100


def classify_image(image_path, modified, h, clf):
    """Classifies passed image using MLPH sub-histogram features and classifier.

    :param image_path: String with path of the image file.
    :param modified: Boolean value to select if the modified MLPH algorithm is
        to be used for histogram computation.
    :param h: Integer with window size
    :param clf: Classifier object
    """

    assert os.path.exists(
        image_path), "Error: Image path entered for clf doesn't exist"

    image_matrix = utils.read_image(image_name=image_path)

    if not modified:
        textures, _, _ = mlph.mlph(data=image_matrix, h=h)
    else:
        textures, _, _ = mlph_modified.mlph_modified(data=image_matrix, h=h)

    textures_list = list(textures.reshape(textures.shape[0] * textures.shape[1],
                                          textures.shape[2]))

    labels = np.array(clf.predict(textures_list))
    labels = labels.reshape((textures.shape[0], textures.shape[1])).T

    clf_image = skimage.color.label2rgb(labels,
                                        colors=['green', 'yellow', 'red',
                                                'blue'])

    start = 'clf_image' if not modified else 'clf_image_modified'
    filename = utils.find_filename_iteratively(start=start,
                                               file_path_suffix='.png',
                                               dir_path='./')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        skimage.io.imsave(filename, clf_image)
