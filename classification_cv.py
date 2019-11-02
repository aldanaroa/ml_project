# Creator: Hoang-Dung Do

import numpy as np
import sklearn.tree
import sklearn.ensemble
import sklearn.model_selection
import scipy.stats


def decision_trees(x_train, y_train, max_depth=10, fold=4, iterations=20):
    dtc = sklearn.tree.DecisionTreeClassifier(random_state=0)
    params = {
        "max_depth": range(1, max_depth + 1)
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(dtc, param_distributions=params, verbose=1, cv=fold,
                                                                  random_state=0, n_iter=iterations)
    print("Training decision tree classifier ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def random_forest(x_train, y_train, max_estimator=100, fold=4, iterations=20):
    rf = sklearn.ensemble.RandomForestClassifier(random_state=0)
    params = {
        "n_estimators": range(1, max_estimator + 1)
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(rf, param_distributions=params, verbose=1, cv=fold,
                                                                  random_state=0, n_iter=iterations)

    print("Training random forest classifier ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def SVC(x_train, y_train, kernels, C_min, C_max, gamma_min, gamma_max, fold=4, iterations=20):
    svc = sklearn.svm.SVC(random_state=0)
    params = {
        "C": scipy.stats.reciprocal(C_min, C_max),
        "gamma": scipy.stats.reciprocal(gamma_min, gamma_max),
        "kernel": kernels
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(svc, param_distributions=params, verbose=1, cv=fold,
                                                                  random_state=0, n_iter=iterations)

    print("Training SVC ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def KNC(x_train, y_train, neighbors=10, fold=4, iterations=20):
    knc = sklearn.neighbors.KNeighborsClassifier(random_state=0)
    params = {
        "n_neighbors": range(1, neighbors + 1),
        "weights": ['uniform', 'distance']
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(knc, param_distributions=params, verbose=1, cv=fold,
                                                                  random_state=0, n_iter=iterations)

    print("Training KNC ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def logistic_regression(x_train, y_train, C_min, C_max, fold=4, iterations=20):
    lrc = sklearn.linear_model.LogisticRegression(random_state=0)
    params = {
        "penalty": ['l1', 'l2', 'elasticnet', 'none'],
        "C": scipy.stats.reciprocal(C_min, C_max),
        "solver": ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(lrc, param_distributions=params, verbose=1, cv=fold,
                                                                  random_state=0, n_iter=iterations)

    print("Training Logistic Regression ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def ada_boost_classifier(x_train, y_train, estimator, no_estimators, fold=4, iterations=20):
    ada_boost = sklearn.ensemble.AdaBoostClassifier(random_state=0, base_estimator=estimator)
    params = {
        "n_estimators": range(1, no_estimators + 1),
        "algorithm": ['SAMME', 'SAMME.R']
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(ada_boost, param_distributions=params, verbose=1,
                                                                  cv=fold,
                                                                  random_state=0, n_iter=iterations)

    print("Training Logistic Regression ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_
