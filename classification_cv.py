# Creator: Hoang-Dung Do

import math
import numpy as np
import sklearn.tree
import sklearn.ensemble
import sklearn.model_selection
import sklearn.neural_network
import sklearn.naive_bayes
import scipy.stats
import timeout_decorator


def decision_trees(x_train, y_train, max_depth=10, fold=4, iterations=20):
    dtc = sklearn.tree.DecisionTreeClassifier(random_state=0)
    params = {
        "max_depth": range(1, max_depth + 1)
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(dtc, param_distributions=params, verbose= 0, cv=fold,
                                                                  random_state=0, n_iter=min(max_depth, iterations))
    print("Training decision tree classifier ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def random_forest(x_train, y_train, max_estimator=100, fold=4, iterations=20):
    rf = sklearn.ensemble.RandomForestClassifier(random_state=0)
    params = {
        "n_estimators": range(1, max_estimator + 1)
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(rf, param_distributions=params, verbose=0, cv=fold,
                                                                  random_state=0, n_iter=min(max_estimator, iterations))

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
    #@timeout_decorator.timeout(460) #test
    random_search_cv = sklearn.model_selection.RandomizedSearchCV(svc, param_distributions=params, verbose=3, cv=fold,
                                                                  random_state=0, n_iter=iterations, error_score = 0)

    print("Training SVC ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def KNC(x_train, y_train, neighbors=10, fold=4, iterations=20):
    knc = sklearn.neighbors.KNeighborsClassifier()
    params = {
        "n_neighbors": range(1, neighbors + 1),
        "weights": ['uniform', 'distance']
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(knc, param_distributions=params, verbose=0, cv=fold,
                                                                  random_state=0, n_iter=iterations)

    print("Training KNC ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def logistic_regression(x_train, y_train, C_min, C_max, fold=4, iterations=20):
    lrc = sklearn.linear_model.LogisticRegression(random_state=0, solver='lbfgs')
    params = {
        "C": scipy.stats.reciprocal(C_min, C_max),
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(lrc, param_distributions=params, verbose=0, cv=fold,
                                                                  random_state=0, n_iter=iterations)

    print("Training Logistic Regression ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def ada_boost_classifier(x_train, y_train, no_estimators=10, fold=4, iterations=20):
    ada_boost = sklearn.ensemble.AdaBoostClassifier(random_state=0)
    params = {
        "n_estimators": range(1, max(no_estimators, 1) + 1),
        "algorithm": ['SAMME', 'SAMME.R']
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(ada_boost, param_distributions=params, verbose=0,
                                                                  cv=fold, random_state=0, n_iter=iterations)

    print("Training Ada Boost ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def GaussianNB(x_train, y_train, fold=4, iterations=20):
    nb = sklearn.naive_bayes.GaussianNB()
    params = {
        "var_smoothing": scipy.stats.reciprocal(math.exp(-10), math.exp(-8))
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(nb, param_distributions=params, verbose=0,
                                                                  cv=fold, random_state=0, n_iter=iterations)

    print("Training Gaussian Naive Bayes ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_


def MLPClassifier(x_train, y_train,
                  hidden_layer_sizes,
                  alphas, max_iter,
                  fold=4, iterations=20):
    nn = sklearn.neural_network.MLPClassifier(random_state=0,
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
