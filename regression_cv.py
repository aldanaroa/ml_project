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

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def decision_tree(x_train, y_train, max_depth=10, fold=4, iterations=20):
    dtc = sklearn.tree.DecisionTreeRegressor(random_state=0)
    params = {
        "max_depth": range(1, max_depth + 1)
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(dtc, param_distributions=params, verbose=0, cv=fold,
                                                                  random_state=0, n_iter=min(max_depth, iterations))
    print("Training decision tree regression ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def random_forest(x_train, y_train, max_estimator=100, fold=4, iterations=20):
    rf = sklearn.ensemble.RandomForestRegressor(random_state=0)
    params = {
        "n_estimators": range(1, max_estimator + 1)
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(rf, param_distributions=params, verbose=0, cv=fold,
                                                                  random_state=0, n_iter=min(max_estimator, iterations))

    print("Training random forest regression ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def SVR(x_train, y_train, kernels, C_min, C_max, gamma_min, gamma_max, fold=4, iterations=20):
    svc = sklearn.svm.SVR()
    params = {
        "C": scipy.stats.reciprocal(C_min, C_max),
        "gamma": scipy.stats.reciprocal(gamma_min, gamma_max),
        "kernel": kernels
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(svc, param_distributions=params, verbose=0, cv=fold,
                                                                  random_state=0, n_iter=iterations)

    print("Training SVR ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def ada_boost_regression(x_train, y_train, estimators, no_estimators, fold=4, iterations=20):
    ada_boost = sklearn.ensemble.AdaBoostRegressor(random_state=0)
    params = {
        "base_estimator": estimators,
        "n_estimators": range(1, no_estimators + 1)
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(ada_boost, param_distributions=params, verbose=0,
                                                                  cv=fold, random_state=0, n_iter=iterations)

    print("Training Ada Boost ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def NeuralNetworkRegression(x_train, y_train,
                            hidden_layer_sizes,
                            alphas, max_iter,
                            fold=4, iterations=20):
    nn = sklearn.neural_network.MLPRegressor(random_state=0,
                                             solver='sgd',
                                             batch_size=int(x_train.shape[0] / 50),
                                             learning_rate_init=0.01,
                                             momentum=0.9,
                                             max_iter=max_iter)
    params = {
        "hidden_layer_sizes": hidden_layer_sizes,
        "alpha": alphas,
        "max_iter": max_iter
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(nn, param_distributions=params, verbose=0,
                                                                  cv=fold, random_state=0, n_iter=iterations)

    print("Training Neural Networks ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_
