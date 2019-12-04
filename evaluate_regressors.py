# Creator: Hoang-Dung Do

import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import json
import regression_cv
import data_processing
import constant
from scipy import stats


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


def evaluate_regressor2(x_train, x_test, y_train, y_test, model, params, dataset_name, model_name):
    y_train_pred = model.predict(x_train)
    y_pred = model.predict(x_test)

    train_r = stats.pearsonr(y_train, y_train_pred)
    test_r = stats.pearsonr(y_test, y_pred)

    print("--{0}:".format(model_name))
    print("\tTraining Pearsonr: {0:.10f}".format(train_r[0]))
    print("\tTesting Pearsonr: {0:.10f}".format(test_r[0]))
    if params is not None and len(params.keys()) > 0:
        print("\tHyperparam:")
        for hyperparam in params.keys():
            print("\t\t {0}: {1}".format(hyperparam, params[hyperparam]))

    return {
        constant.PEARSON: test_r[0]
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

    # Random Forest
    lr_best_model, lr_params = regression_cv.random_forest(x_train, y_train, max_estimator=100, fold=3, iterations=20)
    result[constant.RANDOM_FOREST] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                                        "Wine quality", "Random Forest")

    # SVM
    lr_best_model, lr_params = regression_cv.SVR(x_train, y_train,
                                                 ['linear', 'poly', 'rbf', 'sigmoid'],
                                                 0.01, 100, 1, 1000,
                                                 fold=3, iterations=20)
    result[constant.SVC] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                              "Wine quality", "SVM")

    # AdaBoost
    lr_best_model, lr_params = regression_cv.ada_boost_regression(x_train, y_train, no_estimators=50,
                                                                  fold=3, iterations=20)
    result[constant.ADABOOST] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                                   "Wine quality", "AdaBoost")

    # NeuralNet
    lr_best_model, lr_params = regression_cv.NeuralNetworkRegression(x_train, y_train,
                                                                     hidden_layer_sizes=[(), (10,)],
                                                                     alphas=[0.01, 0.05, 0.5, 1],
                                                                     max_iter=[10, 20, 50, 100],
                                                                     fold=3, iterations=20)
    result[constant.NN] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                             "Wine quality", "NeuralNet")

    export_result(result, "result/red_wine.json")


def QSAR():
    print("Started training linear regressor on QSAR aquatic toxicity data set.")

    x_train, x_test, y_train, y_test = data_processing.QSAR()
    result = {}

    # LR
    lr_best_model, lr_params = regression_cv.linear_regression(x_train, y_train, fold=3, iterations=20)
    result[constant.LR] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                             "QSAR", "Linear Regression")

    # decision tree
    lr_best_model, lr_params = regression_cv.decision_tree(x_train, y_train, max_depth=20, fold=3, iterations=20)
    result[constant.DECISION_TREE] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                                        "QSAR", "Decision tree")

    # Random Forest
    lr_best_model, lr_params = regression_cv.random_forest(x_train, y_train, max_estimator=100, fold=3, iterations=20)
    result[constant.RANDOM_FOREST] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                                        "QSAR", "Random Forest")

    # SVM
    lr_best_model, lr_params = regression_cv.SVR(x_train, y_train,
                                                 ['linear', 'poly', 'rbf', 'sigmoid'],
                                                 0.01, 100, 1, 1000,
                                                 fold=3, iterations=20)
    result[constant.SVC] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                              "QSAR", "SVM")

    # AdaBoost
    lr_best_model, lr_params = regression_cv.ada_boost_regression(x_train, y_train, no_estimators=50,
                                                                  fold=3, iterations=20)
    result[constant.ADABOOST] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                                   "QSAR", "AdaBoost")

    # NeuralNet
    lr_best_model, lr_params = regression_cv.NeuralNetworkRegression(x_train, y_train,
                                                                     hidden_layer_sizes=[(), (10,)],
                                                                     alphas=[0.01, 0.05, 0.5, 1],
                                                                     max_iter=[10, 20, 50, 100],
                                                                     fold=3, iterations=20)
    result[constant.NN] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                             "QSAR", "NeuralNet")

    export_result(result, "result/regression/qsar.json")


def bike():
    print("Started training linear regressor on bike data set.")
    x_train, x_test, y_train, y_test = data_processing.bike()
    result = {}

    # LR
    # lr_best_model, lr_params = regression_cv.linear_regression(x_train, y_train, fold=3, iterations=20)
    # result[constant.LR] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
    #                                         "Bike users", "Linear Regression")

    # decision tree
    # lr_best_model, lr_params = regression_cv.decision_tree(x_train, y_train, max_depth=20, fold=3, iterations=20)
    # result[constant.DECISION_TREE] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
    #                                                    "Bike users", "Decision tree")

    # Random Forest
    # lr_best_model, lr_params = regression_cv.random_forest(x_train, y_train, max_estimator=100, fold=3, iterations=20)
    # result[constant.RANDOM_FOREST] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
    #                                                    "Bike users", "Random Forest")

    # SVM
    lr_best_model, lr_params = regression_cv.SVR(x_train, y_train,
                                                 # ['linear', 'poly', 'rbf', 'sigmoid'],
                                                 ['poly'],
                                                 0.01, 100, 1, 1000,
                                                 fold=3, iterations=1)
    result[constant.SVC] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                              "Bike users", "SVM")

    # AdaBoost
    # lr_best_model, lr_params = regression_cv.ada_boost_regression(x_train, y_train, no_estimators=50,
    #                                                              fold=3, iterations=20)
    # result[constant.ADABOOST] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
    #                                               "Bike users", "AdaBoost")

    # NeuralNet
    lr_best_model, lr_params = regression_cv.NeuralNetworkRegression(x_train, y_train,
                                                                     hidden_layer_sizes=[(10,)],
                                                                     alphas=[0.01, 0.05, 0.5, 1],
                                                                     max_iter=[10, 20, 50, 100],
                                                                     fold=3, iterations=20)
    result[constant.NN] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                             "Bike users", "NeuralNet")

    export_result(result, "result/bike.json")


def student():
    print("Started training linear regressor on student data set.")
    x_train, x_test, y_train, y_test = data_processing.student()
    result = {}

    # LR
    lr_best_model, lr_params = regression_cv.linear_regression(x_train, y_train, fold=3, iterations=20)
    result[constant.LR] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                             "Student Grade", "Linear Regression")

    # decision tree
    lr_best_model, lr_params = regression_cv.decision_tree(x_train, y_train, max_depth=20, fold=3, iterations=20)
    result[constant.DECISION_TREE] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                                        "Student Grade", "Decision tree")

    # Random Forest
    lr_best_model, lr_params = regression_cv.random_forest(x_train, y_train, max_estimator=100, fold=3, iterations=20)
    result[constant.RANDOM_FOREST] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                                        "Student Grade", "Random Forest")

    # SVM
    lr_best_model, lr_params = regression_cv.SVR(x_train, y_train,
                                                 # ['linear', 'poly', 'rbf', 'sigmoid'],
                                                 ['sigmoid'],
                                                 0.01, 100, 1, 1000,
                                                 fold=3, iterations=10)
    result[constant.SVC] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                              "Student Grade", "SVM")

    # AdaBoost
    lr_best_model, lr_params = regression_cv.ada_boost_regression(x_train, y_train, no_estimators=50,
                                                                  fold=3, iterations=20)
    result[constant.ADABOOST] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                                   "Student Grade", "AdaBoost")

    # NeuralNet
    lr_best_model, lr_params = regression_cv.NeuralNetworkRegression(x_train, y_train,
                                                                     hidden_layer_sizes=[(10,), (100, 5,)],
                                                                     alphas=[0.01, 0.05, 0.5, 1],
                                                                     max_iter=[10, 20, 50, 100],
                                                                     fold=3, iterations=20)
    result[constant.NN] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                             "Student Grade", "NeuralNet")

    export_result(result, "result/student.json")


def concrete():
    print("Started training linear regressor on concrete data set.")
    x_train, x_test, y_train, y_test = data_processing.concrete()
    result = {}

    # LR
    lr_best_model, lr_params = regression_cv.linear_regression(x_train, y_train, fold=3, iterations=20)
    result[constant.LR] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                             "concrete", "Linear Regression")

    # decision tree
    lr_best_model, lr_params = regression_cv.decision_tree(x_train, y_train, max_depth=20, fold=3, iterations=20)
    result[constant.DECISION_TREE] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                                        "concrete", "Decision tree")

    # Random Forest
    lr_best_model, lr_params = regression_cv.random_forest(x_train, y_train, max_estimator=100, fold=3, iterations=20)
    result[constant.RANDOM_FOREST] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                                        "concrete", "Random Forest")

    # SVM
    lr_best_model, lr_params = regression_cv.SVR(x_train, y_train,
                                                 # ['linear', 'poly', 'rbf', 'sigmoid'],
                                                 ['linear'],
                                                 0.01, 100, 1, 1000,
                                                 fold=3, iterations=10)
    result[constant.SVC] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                              "concrete", "SVM")

    # AdaBoost
    lr_best_model, lr_params = regression_cv.ada_boost_regression(x_train, y_train, no_estimators=50,
                                                                  fold=3, iterations=20)
    result[constant.ADABOOST] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                                   "concrete", "AdaBoost")

    # NeuralNet
    lr_best_model, lr_params = regression_cv.NeuralNetworkRegression(x_train, y_train,
                                                                     hidden_layer_sizes=[(100,), (10,)],
                                                                     alphas=[0.01, 0.05, 0.5, 1],
                                                                     max_iter=[10, 20, 50, 100],
                                                                     fold=3, iterations=20)
    result[constant.NN] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                             "concrete", "NeuralNet")

    export_result(result, "result/concrete.json")


def gpu():
    print("Started training linear regressor on GPU data set.")
    x_train, x_test, y_train, y_test = data_processing.gpu()
    result = {}

    # LR
    # lr_best_model, lr_params = regression_cv.linear_regression(x_train, y_train, fold=3, iterations=20)
    # result[constant.LR] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
    #                                         "GPU", "Linear Regression")

    # decision tree
    # lr_best_model, lr_params = regression_cv.decision_tree(x_train, y_train, max_depth=20, fold=3, iterations=20)
    # result[constant.DECISION_TREE] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
    #                                                    "GPU", "Decision tree")

    # Random Forest
    # lr_best_model, lr_params = regression_cv.random_forest(x_train, y_train, max_estimator=100, fold=3, iterations=20)
    # result[constant.RANDOM_FOREST] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
    #                                                    "GPU", "Random Forest")

    # SVM
    # lr_best_model, lr_params = regression_cv.SVR(x_train, y_train,
    # ['linear', 'poly', 'rbf', 'sigmoid'],
    #                                             ['linear'],
    #                                             0.01, 100, 1, 1000,
    #                                             fold=3, iterations=10)
    # result[constant.SVC] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
    #                                          "GPU", "SVM")

    # AdaBoost
    lr_best_model, lr_params = regression_cv.ada_boost_regression(x_train, y_train, no_estimators=50,
                                                                  fold=3, iterations=20)
    result[constant.ADABOOST] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                                   "GPU", "AdaBoost")

    # NeuralNet
    lr_best_model, lr_params = regression_cv.NeuralNetworkRegression(x_train, y_train,
                                                                     hidden_layer_sizes=[(100,), (10,)],
                                                                     alphas=[0.01, 0.05, 0.5, 1],
                                                                     max_iter=[10, 20, 50, 100],
                                                                     fold=3, iterations=20)
    result[constant.NN] = evaluate_regressor(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                             "GPU", "NeuralNet")

    export_result(result, "result/GPU.json")


######################################################################################

def molecular():
    print("Started training linear regressor on molecular data set.")
    x_train, x_test, y_train, y_test, x2_train, x2_test, y2_train, y2_test = data_processing.molecular()
    result = {}

    # LR
    lr_best_model, lr_params = regression_cv.linear_regression(x_train, y_train, fold=3, iterations=20)
    evaluate_regressor2(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                        "molecular", "Linear Regression")

    lr_best_model, lr_params = regression_cv.linear_regression(x2_train, y2_train, fold=3, iterations=20)
    result[constant.LR] = evaluate_regressor2(x2_train, x2_test, y2_train, y2_test, lr_best_model, lr_params,
                                              "molecular", "Linear Regression")

    # decision tree
    # lr_best_model, lr_params = regression_cv.decision_tree(x_train, y_train, max_depth=20, fold=3, iterations=20)
    # result[constant.DECISION_TREE] = evaluate_regressor2(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
    #                                                    "molecular", "Decision tree")

    # Random Forest
    # lr_best_model, lr_params = regression_cv.random_forest(x_train, y_train, max_estimator=100, fold=3, iterations=20)
    # result[constant.RANDOM_FOREST] = evaluate_regressor2(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
    #                                                    "molecular", "Random Forest")

    # SVM
    # lr_best_model, lr_params = regression_cv.SVR(x_train, y_train,
    # ['linear', 'poly', 'rbf', 'sigmoid'],
    #                                             ['linear'],
    #                                             0.01, 100, 1, 1000,
    #                                             fold=3, iterations=10)
    # result[constant.SVC] = evaluate_regressor2(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
    #                                          "molecular", "SVM")

    # AdaBoost
    # lr_best_model, lr_params = regression_cv.ada_boost_regression(x_train, y_train, no_estimators=50,
    #                                                              fold=3, iterations=20)
    # result[constant.ADABOOST] = evaluate_regressor2(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
    #                                               "molecular", "AdaBoost")

    # NeuralNet
    # lr_best_model, lr_params = regression_cv.NeuralNetworkRegression(x_train, y_train,
    #                                                                 hidden_layer_sizes=[(100,),(10,)],
    #                                                                 alphas=[0.01, 0.05, 0.5, 1],
    #                                                                 max_iter=[10, 20, 50, 100],
    #                                                                 fold=3, iterations=20)
    # result[constant.NN] = evaluate_regressor2(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
    #                                         "molecular", "NeuralNet")

    export_result(result, "result/molecular.json")
