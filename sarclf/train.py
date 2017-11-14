"""Functions to train SAR image classifier based on MLPH histogram computations.
"""

from __future__ import division
from __future__ import print_function

from sarclf import mlph
from sarclf import mlph_modified
from sarclf import utils

import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

import os.path


#
# Folder Name to Folder Num Mapping:
# Water: 0, Woodland: 1, Farmland: 2, Building: 3
#


def get_train_image_matrices(folder_name, num_images=4):
    """Gets image matrices for training images.

    :param folder_name: String with name of training image folder in
        input_data/train_images directory path.
    :param num_images: Integer with number of images.
    :return: Matrices from training images.
    """

    image_matrices = []
    path = './input_data/train_images/' + folder_name + '/'

    for image_num in range(4, 4 + num_images):
        image_name = path + str(image_num) + '.tif'
        image_matrices.append(utils.read_image(image_name=image_name))

    return image_matrices


def get_folder_train_data(modified, folder_num, folder_name, h):
    """Creates training data from all labeled image pixels in given folder.

    :param modified: Boolean value to indicate if modified MLPH is used
    :param folder_num: String with folder number
    :param folder_name: String with name of folder
    :param h: Integer with window size for MLPH histogram computations
    :return: training_textures, target_labels numpy matrices
    """

    image_matrices = get_train_image_matrices(folder_name=folder_name)
    training_textures = []

    for image_matrix in image_matrices:
        if not modified:
            textures, _, _ = mlph.mlph(data=image_matrix, h=h)
        else:
            textures, _, _ = mlph_modified.mlph_modified(data=image_matrix, h=h)

        textures = list(textures.reshape(textures.shape[0] * textures.shape[1],
                                         textures.shape[2]))
        training_textures += textures

    n_samples = len(training_textures)
    target_labels = [folder_num] * n_samples
    print('\n***** Completed making data for %s folder *****\n' % folder_name)

    return training_textures, target_labels


def make_mlph_data(modified, h):
    """Creates features and target labels data with MLPH histogram computations.

    :param modified: Boolean value to indicate if modified MLPH is used
    :param h: Integer with window size for MLPH histogram computations
    :return: Training set features and target labels numpy matrices
    """

    folder_list = ['water', 'woodland', 'farmland', 'building']
    X_train = []
    y_train = []

    for folder_num, folder in enumerate(folder_list):
        X_folder, y_folder = get_folder_train_data(modified=modified,
                                                   folder_num=folder_num,
                                                   folder_name=folder, h=h)
        X_train += X_folder
        y_train += y_folder

    make_csv(X_train, y_train, modified=modified)
    save_train_as_npy(X_train, y_train, modified=modified)

    return X_train, y_train


def make_csv(X, y, modified):
    """Creates CSV file with training features and target labels numpy matrices.

    :param X: Numpy matrix with training set features
    :param y: Numpy matrix with training set target labels
    :param modified: Boolean value to indicate if modified MLPH is used
    """

    full_train_array = np.hstack((np.array(y).reshape((-1, 1)), np.array(X)))
    file_name = './output_data/train_data.csv' \
        if not modified else './output_data/modified/train_data.csv'

    np.savetxt(file_name, full_train_array, delimiter=',', fmt='%d')


def save_train_as_npy(X, y, modified):
    """Saves training set features and target labels numpy matrices to file.

    :param X: Numpy matrix with training set features
    :param y: Numpy matrix with training set target labels
    :param modified: Boolean value to indicate if modified MLPH is used
    """

    save_file_name = './output_data/X_train.npy' \
        if not modified else './output_data/modified/X_train.npy'
    np.save(save_file_name, X)

    save_file_name = './output_data/y_train.npy' \
        if not modified else './output_data/modified/y_train.npy'
    np.save(save_file_name, y)


def run_mlph(modified, h):
    """Runs MLPH computations for image to build features training data.

    :param modified: Boolean value to indicate if modified MLPH is used
    :param h: Integer with window size for MLPH histogram computations
    :return: training set features, test set features, training set target
        labels, test set target labels
    """

    X_train, y_train = make_mlph_data(modified=modified, h=h)
    y_train = list(y_train)
    X_train, X_test, y_train, y_test = \
        train_test_split(X_train, y_train, test_size=0.20, random_state=0)

    return X_train, X_test, y_train, y_test


def train_svm(X_train, y_train, modified):
    """Trains and saves SVM classifier based on computed MLPH histograms.

    :param X_train: Numpy matrix with training set features
    :param y_train: Numpy matrix with training set target labels
    :param modified: Boolean value to indicate if modified MLPH is used
    :return: Trained SVM classifier object
    """

    print("\nSVM training started.")

    param_grid = [{'C': np.arange(0.05, 7, 1.5)}]
    score = 'accuracy'

    clf = GridSearchCV(LinearSVC(), param_grid, cv=5, scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Completed SVM training.\n")
    print("Best parameters set found on development set: {}".format(
        clf.best_params_))

    file_name = "./output_data/trained_svm.pickle" \
        if not modified else "./output_data/modified/trained_svm.pickle"

    with open(file_name, "wb") as handle:
        pickle.dump(clf, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return clf


def load_training_data(modified):
    """Loads saved training data from numpy matrix files

    :param modified: Boolean value to indicate if modified MLPH is used
    :return: training set features, test set features, training set target
        labels, test set target labels
    """

    file_name_1 = './output_data/X_train.npy' \
        if not modified else './output_data/modified/X_train.npy'
    file_name_2 = './output_data/y_train.npy' \
        if not modified else './output_data/modified/y_train.npy'

    assert (
        os.path.exists(file_name_1) and os.path.exists(file_name_2)), \
        'Error: run_mlph option off and no saved training data found.'

    X_train = np.load(file_name_1)
    y_train = np.load(file_name_2)
    y_train = list(y_train)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train,
                                                        test_size=0.20,
                                                        random_state=42)

    return X_train, X_test, y_train, y_test


def load_svm(modified):
    """Loads trained SVM classifier from pickle.

    :param modified: Boolean value to indicate if modified MLPH is used
    :return: Loaded SVM classifier
    """

    file_name = "./output_data/trained_svm.pickle" \
        if not modified else "./output_data/modified/trained_svm.pickle"

    assert (os.path.exists(
        file_name)), "Error: load_svm option is off and no saved svm found."

    with open(file_name, "rb") as handle:
        clf = pickle.load(handle)

    return clf
