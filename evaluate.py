# Creator: Hoang-Dung Do

import sklearn.metrics
import classification_cv
import regression_cv
import data_processing


def diabete_retinopathy():
    print("Started training classifiers on diabete retinopathy data set.")
    x_train, x_test, y_train, y_test = data_processing.diabetic_retinopathy()

    # DECISION TREE
    ct_best_model, ct_params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=4, iterations=20)
    y_predict_dt = ct_best_model.predict(x_test)
    dt_accuracy = sklearn.metrics.accuracy_score(y_test, y_predict_dt)
    print("--DECISION TREE:")
    print("\tDiabetic retinopathy accuracy: {0:.2f}%".format(dt_accuracy * 100))
    print("\tHyperparam max_depth: {0}".format(ct_params['max_depth']))

    # RANDOM FOREST
    rf_best_model, rf_params = classification_cv.random_forest(x_train, y_train, max_estimator=100, fold=4,
                                                               iterations=20)
    y_predict_rf = rf_best_model.predict(x_test)
    rf_accuracy = sklearn.metrics.accuracy_score(y_test, y_predict_rf)
    print("--RANDOM FOREST:")
    print("\tDiabetic retinopathy accuracy: {0:.2f}%".format(rf_accuracy * 100))
    print("\tHyperparam n_estimators: {0}".format(rf_params['n_estimators']))

    # SVM
    svc_best_model, svc_params = classification_cv.SVC(x_train, y_train, ['linear', 'rbf'], 0.01, 10, 1, 1000, fold=4,
                                                       iterations=20)
    y_predict_svc = svc_best_model.predict(x_test)
    svc_accuracy = sklearn.metrics.accuracy_score(y_test, y_predict_svc)
    print("--SVM:")
    print("\tDiabetic retinopathy accuracy: {0:.2f}%".format(svc_accuracy * 100))
    print("\tHyperparams n_estimators: {0}".format(svc_params['n_estimators']))
    print("\tkernel: {0}".format(svc_params['kernel']))
    print("\tC: {0}".format(svc_params['C']))
    print("\tgamma: {0}".format(svc_params['gamma']))


def default_credit_card():
    print("Started training classifiers on default credit card clients data set.")
    x_train, x_test, y_train, y_test = data_processing.credit_card_client()

    best_model, params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=4, iterations=20)

    y_predict = best_model.predict(x_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_predict)
    print("--Decision tree:")
    print("\tDefault credit card clients accuracy: {0:.2f}%".format(accuracy * 100))
    print("\tHyperparam max_depth: {0}".format(params['max_depth']))


def red_wine_quality():
    print("Started training linear regressor on red wine quality data set.")
    x_train, x_test, y_train, y_test = data_processing.red_wine_quality()

    best_model = regression_cv.linear_regression(x_train, y_train, fold=4, iterations=20)

    y_predict = best_model.predict(x_test)
    accuracy = sklearn.metrics.r2_score(y_test, y_predict)
    print("--Linear Regression:")
    print("\tWine quality prediction accuracy: {0:.2f}%".format(accuracy * 100))


red_wine_quality()
