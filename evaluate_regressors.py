# Creator: Hoang-Dung Do

import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import json
import regression_cv
import data_processing
import constant


def export_result(result, filename):
    with open(filename, 'w') as outfile:
        json.dump(result, outfile)


def evaluate_regressor(x_train, x_test, y_train, y_test, model, params, dataset_name, model_name):
    y_train_pred = model.predict(x_train)
    y_pred = model.predict(x_test)

    train_r2 = sklearn.metrics.r2_score(y_true=y_train, y_pred=y_train_pred)
    test_r2 = sklearn.metrics.r2_score(y_true=y_test, y_pred=y_pred)

    print("--{0}:".format(model_name))
    print("\tTraining R^2: {0:.10f}".format(train_r2))
    print("\tTesting R^2: {0:.10f}".format(test_r2))
    if params is not None and len(params.keys()) > 0:
        print("\tHyperparam:")
        for hyperparam in params.keys():
            print("\t\t {0}: {1}".format(hyperparam, params[hyperparam]))

    return {
        constant.R2: test_r2
    }


def red_wine_quality():
    print("Started training linear regressor on red wine quality data set.")
    x_train, x_test, y_train, y_test = data_processing.red_wine_quality()
    result = {}

    # LR
    lr_best_model, lr_params = regression_cv.linear_regression(x_train, y_train, fold=3, iterations=20)
    result[constant.LR] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                             "Wine quality", "Linear Regression")

    # decision tree
    lr_best_model, lr_params = regression_cv.decision_tree(x_train, y_train, max_depth=20, fold=3, iterations=20)
    result[constant.DECISION_TREE] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                                        "Wine quality", "Decision tree")

    export_result(result, "result/red_wine.json")


red_wine_quality()
