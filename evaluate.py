# Creator: Hoang-Dung Do

import sklearn.metrics
import classification_cv
import regression_cv
import data_processing


def evaluate_classifier(x_train, x_test, y_train, y_test, model, params, dataset_name, model_name):
    y_predict = model.predict(x_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_predict)
    print("--{0}:".format(model_name))
    print("\t{0} accuracy: {1:.2f}%".format(dataset_name, accuracy * 100))
    if params is not None and len(params.keys()) > 0:
        print("\tHyperparam:")
        for hyperparam in params.keys():
            print("\t\t {0}: {1}".format(hyperparam, params[hyperparam]))


def diabete_retinopathy():
    print("Started training classifiers on diabete retinopathy data set.")
    x_train, x_test, y_train, y_test = data_processing.diabetic_retinopathy()

    # DECISION TREE
    dt_best_model, dt_params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=4, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, dt_best_model, dt_params, "Diabete retinopathy",
                        "DECISION TREE")

    # RANDOM FOREST
    rf_best_model, rf_params = classification_cv.random_forest(x_train, y_train, max_estimator=100, fold=4,
                                                               iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params, "Diabete retinopathy",
                        "RANDOM FOREST")

    # SVC
    svc_best_model, svc_params = classification_cv.SVC(x_train, y_train, ['linear', 'rbf'], 0.01, 10, 1, 1000, fold=4,
                                                       iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, svc_best_model, svc_params, "Diabete retinopathy",
                        "SVC")


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


diabete_retinopathy()
