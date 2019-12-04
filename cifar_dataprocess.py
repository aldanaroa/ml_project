# Creator: Hoang-Dung Do

import pickle
import numpy as np
import sklearn.preprocessing
import sklearn.metrics
import sklearn.utils
import sklearn.model_selection


def load_data_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return np.array(dict[b'data']), np.array(dict[b'labels'])


def get_labels():
    with open("data/cifar/batches.meta", 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b"label_names"]


def load_data():
    x1, y1 = load_data_batch("data/cifar/data_batch_1")
    x2, y2 = load_data_batch("data/cifar/data_batch_2")
    x3, y3 = load_data_batch("data/cifar/data_batch_3")
    x4, y4 = load_data_batch("data/cifar/data_batch_4")
    x5, y5 = load_data_batch("data/cifar/data_batch_5")

    x_train_ = np.vstack([x1, x2, x3, x4, x5])
    y_train_ = np.hstack([y1, y2, y3, y4, y5])

    scaler_ = sklearn.preprocessing.StandardScaler()
    x_train_ = scaler_.fit_transform(x_train_)

    x_test_, y_test_ = load_data_batch("data/cifar/test_batch")
    x_test_ = scaler_.transform(x_test_)

    return x_train_, x_test_, y_train_, y_test_


def vertical_flip(x):
    """
    :param x:
    :return:
    """
    flip = np.copy(x)
    return np.flip(flip, -1)


def crop(x):
    """
    :param x:
    :return:
    """
    crop_array = np.copy(x)
    crop_array[:, :, :1, :] = 0
    crop_array[:, :, :, :1] = 0
    crop_array[:, :, -1:, :] = 0
    crop_array[:, :, :, -1:] = 0

    return crop_array


def aug_split():
    x_train_total, x_test, y_train_total, y_test = load_data()

    x_train_total = x_train_total.reshape((-1, 3, 32, 32))
    x_test = x_test.reshape((-1, 3, 32, 32))

    x_train_total = np.vstack((x_train_total, vertical_flip(x_train_total), crop(x_train_total)))
    y_train_total = np.hstack((y_train_total, y_train_total, y_train_total)).reshape(-1)

    x_train_total, y_train_total = sklearn.utils.shuffle(x_train_total, y_train_total, random_state=0)
    x_train, x_cv, y_train, y_cv = sklearn.model_selection.train_test_split(x_train_total, y_train_total,
                                                                            test_size=0.2, random_state=0)

    return x_train, x_cv, y_train, y_cv, x_test, y_test

