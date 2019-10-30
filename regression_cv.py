# Creator: Hoang-Dung Do

import numpy as np
import sklearn.tree
import sklearn.ensemble
import sklearn.linear_model
import sklearn.model_selection
import scipy.stats


def linear_regression(x_train, y_train, fold=4, iterations=20):
    ls = sklearn.linear_model.LinearRegression()

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(ls, param_distributions={}, verbose=1, cv=fold,
                                                                  random_state=0, n_iter=iterations)

    print("Training SVR ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_
