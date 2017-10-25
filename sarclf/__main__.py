from __future__ import division
from __future__ import print_function

from sarclf import train
from sarclf import test

import argparse


def main():
    parser = argparse.ArgumentParser(description='IP-mlph-sar_clf-project cmd line options')
    parser.add_argument('-run_mlph', help='0/1: whether to make training data from scratch', type=int, default=0)
    parser.add_argument('-train_svm', help='0/1: whether to make train SVM from scratch', type=int, default=0)
    parser.add_argument('-modified', help='0/1: whether to run modified MLPH algorithm', type=int, default=0)
    parser.add_argument('-h_param', help='value for window size h', type=int, default=5)
    parser.add_argument('-test', help='whether to run test clf on image pixels', type=int, default=1)
    parser.add_argument('-clfimg', help='path of sar image to classify', type=str, default='')

    args = vars(parser.parse_args())
    run_mlph = bool(args['run_mlph'])
    train_svm = bool(args['train_svm'])
    modified_mlph = bool(args['modified'])
    h = args['h_param']
    test_pixel_clf = bool(args['test'])
    image_filename = args['clfimg']

    assert (h % 2 == 1), "h must me an odd integer"

    if run_mlph:
        X_train, X_test, y_train, y_test = train.run_mlph(modified=modified_mlph, h=h)
    else:
        X_train, X_test, y_train, y_test = train.load_training_data(modified=modified_mlph)

    if train_svm:
        clf = train.train_svm(X_train=X_train, y_train=y_train, modified=modified_mlph)
    else:
        clf = train.load_svm(modified=modified_mlph)

    if test_pixel_clf:
        test.test_pixel_classification(X_test=X_test, y_test=y_test, clf=clf)

    if image_filename != '':
        test.classify_image(image_path=image_filename, modified=modified_mlph, h=h, clf=clf)

    return


if __name__ == '__main__':
    main()
