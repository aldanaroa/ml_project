# Creator: Hoang-Dung Do

import sklearn.metrics
import classification_cv
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
    rf_best_model, rf_params = classification_cv.random_forest(x_train, y_train, max_estimator=100, fold=4, iterations=20)
    y_predict_rf = rf_best_model.predict(x_test)
    rf_accuracy = sklearn.metrics.accuracy_score(y_test, y_predict_rf)
    print("--RANDOM FOREST:")
    print("\tDiabetic retinopathy accuracy: {0:.2f}%".format(rf_accuracy * 100))
    print("\tHyperparam n_estimators: {0}".format(rf_params['n_estimators']))


def default_credit_card():
    print("Started training classifiers on default credit card clients data set.")
    x_train, x_test, y_train, y_test = data_processing.credit_card_client()

    best_model, params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=4, iterations=20)

    y_predict = best_model.predict(x_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_predict)
    print("--Decision tree:")
    print("\tDefault credit card clients accuracy: {0:.2f}%".format(accuracy * 100))
    print("\tHyperparam max_depth: {0}".format(params['max_depth']))


diabete_retinopathy()
