# Creator: Hoang-Dung Do

import sklearn.model_selection


def normalize_split(x, y, test_size):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, random_state=0,
                                                                                test_size=test_size)
    scaler = sklearn.preprocessing.StandardScaler()

    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, x_test, y_train, y_test
