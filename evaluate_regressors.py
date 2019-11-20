# Creator: Hoang-Dung Do

import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import regression_cv
import data_processing


def evaluate_regressor(x_train, x_test, y_train, y_test, model, params, dataset_name, model_name):
    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)
    print("--{0}:".format(model_name))
    print("\tTraining accuracy: {0:.2f}%".format(dataset_name, train_accuracy * 100))
    print("\tTesting accuracy: {0:.2f}%".format(dataset_name, test_accuracy * 100))
    if params is not None and len(params.keys()) > 0:
        print("\tHyperparam:")
        for hyperparam in params.keys():
            print("\t\t {0}: {1}".format(hyperparam, params[hyperparam]))
    return train_accuracy, test_accuracy


def red_wine_quality():
    print("Started training linear regressor on red wine quality data set.")
    x_train, x_test, y_train, y_test = data_processing.red_wine_quality()

    # LR
    lr_best_model, lr_params = regression_cv.linear_regression(x_train, y_train, fold=3, iterations=20)
    evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params, "Wine quality", "Linear Regression")

    # decision tree
    lr_best_model, lr_params = regression_cv.decision_tree(x_train, y_train, max_depth=20, fold=3, iterations=20)
    evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params, "Wine quality", "Decision tree")


red_wine_quality()
