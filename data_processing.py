# Creator: Hoang-Dung Do

import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.preprocessing as preprocessing
import data_util
import re


# ===================================CLASSIFICATION============================================ #

def diabetic_retinopathy(test_size=0.2):
    data = np.loadtxt("data/classification/messidor_features.arff", delimiter=',', skiprows=24)
    x, y = data[:, :18], data[:, 18]
    return data_util.normalize_split(x, y, test_size)


def credit_card_client(test_size=0.2):
    data = np.loadtxt("data/classification/default_credit_card_clients.csv", delimiter=',', skiprows=2)
    x, y = data[:, 1:24].astype(int), data[:, 24].astype(int)
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


def clean_1(x):
    return x.strip()


def adult():
    names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
             'race', 'sex',
             'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']
    data = pd.read_csv('data/classification/Adult/adult.data', delimiter=',', header=None, names=names, na_values=['?'])
    test = pd.read_csv('data/classification/Adult/adult.test', delimiter=',', header=None, names=names, skiprows=1,
                       na_values=['?'])

    for ser in data.select_dtypes(include=object):
        data[ser].map(clean_1)
    for ser in test.select_dtypes(include=object):
        test[ser].map(clean_1)

    pattern = r'([^?\s.]{1,20})'  # eliminate spaces(again), ? and period
    re.compile(pattern)
    for ser in data.select_dtypes(include=object):
        data[ser] = data[ser].str.extract(pattern)

    for ser in test.select_dtypes(include=object):
        test[ser] = test[ser].str.extract(pattern)

    data.dropna(inplace=True)
    test.dropna(inplace=True)
    # encode categorical data in data and test
    enc = preprocessing.OrdinalEncoder()
    enc.fit(data.select_dtypes(include=object))
    x_train = enc.transform(data.select_dtypes(include=object))
    x_test = enc.transform(test.select_dtypes(include=object))
    # eight column is the target variable
    y_train = x_train[:, 8]
    y_test = x_test[:, 8]
    # concatenate encoded and scalar data
    x_train = np.concatenate((x_train[:, :8], data.select_dtypes(include=int)), axis=1)
    x_test = np.concatenate((x_test[:, :8], test.select_dtypes(include=int)), axis=1)

    return x_train, x_test, y_train, y_test


def seismic_bumps(test_size=0.2):
    data = np.loadtxt("data/classification/seismic-bumps.arff", delimiter=',', skiprows=155,
                      converters={0: seismic_level, 1: seismic_level, 2: shift, 7: seismic_level})
    x, y = data[:, :18].astype(int), data[:, 18].astype(int)

    return data_util.normalize_split(x, y, test_size)


def seismic_level(char):
    if char == b'a':
        return 0
    if char == b'b':
        return 1
    if char == b'c':
        return 2
    if char == b'd':
        return 3
    return 0


def shift(char):
    if char == b'N':
        return 1
    return 0


def statlog_german(test_size=0.2):
    data = np.loadtxt("data/classification/german.data-numeric", skiprows=0)
    x, y = data[:, :24].astype(int), data[:, 24].astype(int)

    return data_util.normalize_split(x, y, test_size)


# =======================================REGRESSION============================================ #

def red_wine_quality(test_size=0.2):
    data = np.loadtxt("data/regression/wine_quality/winequality-red.csv", delimiter=';', skiprows=1)
    x, y = data[:, :11], data[:, 11]

    return data_util.normalize_split(x, y, test_size)


def community_crime(test_size=0.2):
    used_cols = set(range(0, 128)) - set(range(0, 5)) - set(range(101, 118)) - set(range(121, 127))
    data = np.loadtxt("data/regression/communities.data", delimiter=',', usecols=used_cols)
    x, y = data[:, :99], data[:, 99]

    return data_util.normalize_split(x, y, test_size)


def QSAR(test_size=0.2):
    data = np.loadtxt("data/regression/qsar_aquatic_toxicity.csv", delimiter=';')
    x, y = data[:, :9], data[:, 9]

    return data_util.normalize_split(x, y, test_size)


def facebook_metric(test_size=0.2):
    data = np.loadtxt("data/regression/facebook_metrics/dataset_Facebook.csv", delimiter=';', skiprows=1,
                      converters={1: post_type})
    x, y = data[:, :18], data[:, 18]

    return data_util.normalize_split(x, y, test_size)


def post_type(type):
    if type == b"Photo":
        return 0
    if type == b"Status":
        return 1
    if type == b"Link":
        return 2
    return 3
