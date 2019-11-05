# Creator: Hoang-Dung Do

import numpy as np
import sklearn.model_selection

import data_util


# ===================================CLASSIFICATION============================================ #

def diabetic_retinopathy(test_size=0.2):
    data = np.loadtxt("data/classification/messidor_features.arff", delimiter=',', skiprows=24)
    x, y = data[:, :18], data[:, 18]
    return data_util.normalize_split(x, y, test_size)


def credit_card_client(test_size=0.2):
    data = np.loadtxt("data/classification/default_credit_card_clients.csv", delimiter=',', skiprows=2)
    x, y = data[:, 1:24], data[:, 24]
    return data_util.normalize_split(x, y, test_size)


def breast_cancer(test_size=0.2):
    data = np.loadtxt("data/classification/breast_cancer/wdbc.data",
                      delimiter=',', skiprows=0,
                      converters={1: cancer_type_num})
    x, y = data[:, 2:32], data[:, 1]

    return data_util.normalize_split(x, y, test_size)


def cancer_type_num(char):
    if char == b'M':
        return 1
    return 0


# =======================================REGRESSION============================================ #

def red_wine_quality(test_size=0.2):
    data = np.loadtxt("data/regression/wine_quality/winequality-red.csv", delimiter=';', skiprows=1)
    x, y = data[:, :11], data[:, 11]
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=0,
                                                                                test_size=test_size)

    scaler = sklearn.preprocessing.StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test
