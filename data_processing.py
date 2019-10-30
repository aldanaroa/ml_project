# Creator: Hoang-Dung Do

import numpy as np
import sklearn.model_selection


def diabetic_retinopathy(test_size=0.2):
    data = np.loadtxt("data/classification/messidor_features.arff", delimiter=',', skiprows=24)
    x, y = data[:, :18], data[:, 18]

    scaler = sklearn.preprocessing.StandardScaler()
    x = scaler.fit_transform(x)

    return sklearn.model_selection.train_test_split(x, y, random_state=0, test_size=test_size)


def credit_card_client(test_size=0.2):
    data = np.loadtxt("data/classification/default_credit_card_clients.csv", delimiter=',', skiprows=2)
    x, y = data[:, 1:24], data[:, 24]

    scaler = sklearn.preprocessing.StandardScaler()
    x = scaler.fit_transform(x)

    return sklearn.model_selection.train_test_split(x, y, random_state=0, test_size=test_size)

