# Creator: Hoang-Dung Do

import json
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import pandas as pd
import classification_cv
import regression_cv
import data_processing
import constant


def export_result(result, filename):
    with open(filename, 'w') as outfile:
        json.dump(result, outfile)


def evaluate_classifier(x_train, x_test, y_train, y_test, model, params, dataset_name, model_name):
    y_pred = model.predict(x_test)

    train_accuracy = model.score(x_train, y_train)

    test_accuracy = sklearn.metrics.accuracy_score(y_true=y_test, y_pred=y_pred)
    precision = sklearn.metrics.precision_score(y_true=y_test, y_pred=y_pred)
    recall = sklearn.metrics.recall_score(y_true=y_test, y_pred=y_pred)

    print("--{0}:".format(model_name))
    print("\tTraining accuracy: {1:.2f}%".format(dataset_name, train_accuracy * 100))
    print("\tTesting accuracy: {1:.2f}%".format(dataset_name, test_accuracy * 100))
    print("\tPrecision: {1:.2f}%".format(dataset_name, precision * 100))
    print("\tRecall: {1:.2f}%".format(dataset_name, recall * 100))
    if params is not None and len(params.keys()) > 0:
        print("\tHyperparam:")
        for hyperparam in params.keys():
            print("\t\t {0}: {1}".format(hyperparam, params[hyperparam]))

    return {
        constant.ACCURACY: test_accuracy,
        constant.PRECISION: precision,
        constant.RECALL: recall,
        constant.HYPERPARAM: params
    }


def diabete_retinopathy():
    print("Started training classifiers on diabete retinopathy data set.")

    x_train, x_test, y_train, y_test = data_processing.diabetic_retinopathy()
    result = {}

    # LR
    lr_best_model, lr_params = classification_cv.logistic_regression(x_train, y_train, C_min=0.01, C_max=10, fold=3,
                                                                     iterations=20)
    result[constant.LR] = evaluate_classifier(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                              "Diabete retinopathy", "Logistic Regression")

    # DECISION TREE
    dt_best_model, dt_params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=3, iterations=20)
    result[constant.DECISION_TREE] = evaluate_classifier(x_train, x_test, y_train, y_test, dt_best_model, dt_params,
                                                         "Diabete retinopathy", "DECISION TREE")

    # RANDOM FOREST
    rf_best_model, rf_params = classification_cv.random_forest(x_train, y_train, max_estimator=20, fold=4,
                                                               iterations=20)
    result[constant.RANDOM_FOREST] = evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                                                         "Diabete retinopathy", "RANDOM FOREST")

    # AdaBoost
    rf_best_model, rf_params = classification_cv.ada_boost_classifier(x_train, y_train, no_estimators=50, fold=4,
                                                                      iterations=20)
    result[constant.ADABOOST] = evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                                                    "Diabete retinopathy", "AdaBoost")

    # SVC
    svc_best_model, svc_params = classification_cv.SVC(x_train, y_train,
                                                       ['linear', 'poly', 'rbf', 'sigmoid'],
                                                       0.01, 100, 1, 1000, fold=4,
                                                       iterations=20)
    result[constant.SVC] = evaluate_classifier(x_train, x_test, y_train, y_test, svc_best_model, svc_params,
                                               "Diabete retinopathy", "SVC")

    # KNN
    knn_best_model, knn_params = classification_cv.KNC(x_train, y_train, neighbors=10, fold=4, iterations=20)
    result[constant.KNC] = evaluate_classifier(x_train, x_test, y_train, y_test, knn_best_model, knn_params,
                                               "Diabete retinopathy", "KNN")

    # GaussianNB
    nb_best_model, nb_params = classification_cv.GaussianNB(x_train, y_train, fold=4, iterations=20)
    result[constant.GNB] = evaluate_classifier(x_train, x_test, y_train, y_test, nb_best_model, nb_params,
                                               "Diabete retinopathy", "Gaussian NB")

    # MLP
    mlp_best_model, mlp_params = classification_cv.MLPClassifier(x_train, y_train,
                                                                 hidden_layer_sizes=[(), (5,)],
                                                                 alphas=[0.01, 0.05, 0.5, 1],
                                                                 max_iter=[10, 20, 50, 100],
                                                                 fold=4, iterations=20)
    result[constant.NN] = evaluate_classifier(x_train, x_test, y_train, y_test, mlp_best_model, mlp_params,
                                              "Diabete retinopathy", "MLP")

    export_result(result, "result/classification/diabetic_retinopathy.json")


def default_credit_card():
    print("Started training classifiers on default credit card clients data set.")
    x_train, x_test, y_train, y_test = data_processing.credit_card_client()
    result = {}

    # LR
    lr_best_model, lr_params = classification_cv.logistic_regression(x_train, y_train, C_min=0.01, C_max=10, fold=3,
                                                                     iterations=20)
    result[constant.LR] = evaluate_classifier(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                              "Default credit card", "Logistic Regression")

    # DECISION TREE
    dt_best_model, dt_params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=3, iterations=20)
    result[constant.DECISION_TREE] = evaluate_classifier(x_train, x_test, y_train, y_test, dt_best_model,
                                                         dt_params, "Default credit card", "DECISION TREE")

    # RANDOM FOREST
    rf_best_model, rf_params = classification_cv.random_forest(x_train, y_train, max_estimator=20, fold=4,
                                                               iterations=20)
    result[constant.RANDOM_FOREST] = evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                                                         "Default credit card", "RANDOM FOREST")

    # AdaBoost
    rf_best_model, rf_params = classification_cv.ada_boost_classifier(x_train, y_train, no_estimators=50, fold=4,
                                                                      iterations=20)
    result[constant.ADABOOST] = evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                                                    "Default credit card", "AdaBoost")

    # SVC
    svc_best_model, svc_params = classification_cv.SVC(x_train, y_train,
                                                       ['linear'],
                                                       C_min=0.01, C_max=1,
                                                       gamma_min=1, gamma_max=100,
                                                       fold=3, iterations=10)
    result[constant.SVC] = evaluate_classifier(x_train, x_test, y_train, y_test, svc_best_model, svc_params,
                                               "Default credit card", "SVC")

    # KNN
    knn_best_model, knn_params = classification_cv.KNC(x_train, y_train, neighbors=10, fold=3, iterations=20)
    result[constant.KNC] = evaluate_classifier(x_train, x_test, y_train, y_test, knn_best_model, knn_params,
                                               "Default credit card", "KNN")

    # GaussianNB
    nb_best_model, nb_params = classification_cv.GaussianNB(x_train, y_train, fold=4, iterations=20)
    result[constant.GNB] = evaluate_classifier(x_train, x_test, y_train, y_test, nb_best_model, nb_params,
                                               "Default credit card", "Gaussian NB")

    # MLP
    mlp_best_model, mlp_params = classification_cv.MLPClassifier(x_train, y_train,
                                                                 hidden_layer_sizes=[(), (10,)],
                                                                 alphas=[0.01, 0.05, 0.5, 1],
                                                                 max_iter=[100, 1000, 10000, 100000],
                                                                 fold=4, iterations=20)
    result[constant.NN] = evaluate_classifier(x_train, x_test, y_train, y_test, mlp_best_model, mlp_params,
                                              "Breast cancer", "MLP")

    export_result(result, "result/classification/default_credit_card.json")


def breast_cancer():
    print("Started training classifiers on breast cancer data set.")
    x_train, x_test, y_train, y_test = data_processing.breast_cancer()
    result = {}

    # LR
    lr_best_model, lr_params = classification_cv.logistic_regression(x_train, y_train, C_min=0.01, C_max=10, fold=3,
                                                                     iterations=20)
    result[constant.LR] = evaluate_classifier(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                              "Breast cancer",
                                              "Logistic Regression")

    # DECISION TREE
    dt_best_model, dt_params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=3, iterations=20)
    result[constant.DECISION_TREE] = evaluate_classifier(x_train, x_test, y_train, y_test, dt_best_model, dt_params,
                                                         "Breast cancer",
                                                         "DECISION TREE")

    # RANDOM FOREST
    rf_best_model, rf_params = classification_cv.random_forest(x_train, y_train, max_estimator=20, fold=4,
                                                               iterations=20)
    result[constant.RANDOM_FOREST] = evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                                                         "Breast cancer", "RANDOM FOREST")

    # AdaBoost
    rf_best_model, rf_params = classification_cv.ada_boost_classifier(x_train, y_train, no_estimators=50, fold=4,
                                                                      iterations=20)
    result[constant.ADABOOST] = evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                                                    "Breast cancer", "AdaBoost")

    # SVC
    svc_best_model, svc_params = classification_cv.SVC(x_train, y_train,
                                                       ['linear', 'poly', 'rbf', 'sigmoid'],
                                                       0.01, 100, 1, 1000, fold=4,
                                                       iterations=20)
    result[constant.SVC] = evaluate_classifier(x_train, x_test, y_train, y_test, svc_best_model, svc_params,
                                               "Breast cancer", "SVC")

    # KNN
    knn_best_model, knn_params = classification_cv.KNC(x_train, y_train, neighbors=10, fold=4, iterations=20)
    result[constant.KNC] = evaluate_classifier(x_train, x_test, y_train, y_test, knn_best_model, knn_params,
                                               "Breast cancer", "KNN")

    # GaussianNB
    nb_best_model, nb_params = classification_cv.GaussianNB(x_train, y_train, fold=4, iterations=20)
    result[constant.GNB] = evaluate_classifier(x_train, x_test, y_train, y_test, nb_best_model, nb_params,
                                               "Breast cancer",
                                               "Gaussian NB")

    # MLP
    mlp_best_model, mlp_params = classification_cv.MLPClassifier(x_train, y_train,
                                                                 hidden_layer_sizes=[(), (10,)],
                                                                 alphas=[0.01, 0.05, 0.5, 1],
                                                                 max_iter=[10, 20, 50, 100],
                                                                 fold=4, iterations=20)
    result[constant.NN] = evaluate_classifier(x_train, x_test, y_train, y_test, mlp_best_model, mlp_params,
                                              "Breast cancer", "MLP")

    export_result(result, "result/classification/breast_cancer.json")


def adult():
    print("Started training classifiers on Adult data set.")
    x_train, x_test, y_train, y_test = data_processing.adult()
    result = {}

    # LR
    lr_best_model, lr_params = classification_cv.logistic_regression(x_train, y_train, C_min=0.01, C_max=10, fold=3,
                                                                     iterations=20)
    result[constant.LR] = evaluate_classifier(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                              "Adult",
                                              "Logistic Regression")

    # DECISION TREE
    dt_best_model, dt_params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=3, iterations=20)
    result[constant.DECISION_TREE] = evaluate_classifier(x_train, x_test, y_train, y_test, dt_best_model, dt_params,
                                                         "Adult",
                                                         "DECISION TREE")

    # RANDOM FOREST
    rf_best_model, rf_params = classification_cv.random_forest(x_train, y_train, max_estimator=20, fold=4,
                                                               iterations=20)
    result[constant.RANDOM_FOREST] = evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                                                         "Adult", "RANDOM FOREST")

    # AdaBoost
    rf_best_model, rf_params = classification_cv.ada_boost_classifier(x_train, y_train, no_estimators=50, fold=4,
                                                                      iterations=20)
    result[constant.ADABOOST] = evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                                                    "Adult", "AdaBoost")

    # SVC
    svc_best_model, svc_params = classification_cv.SVC(x_train, y_train,
                                                       ['linear', 'poly', 'rbf', 'sigmoid'],
                                                       0.01, 100, 1, 1000, fold=4,
                                                       iterations=20)
    result[constant.SVC] = evaluate_classifier(x_train, x_test, y_train, y_test, svc_best_model, svc_params,
                                               "Adult", "SVC")

    # KNN
    knn_best_model, knn_params = classification_cv.KNC(x_train, y_train, neighbors=10, fold=4, iterations=20)
    result[constant.KNC] = evaluate_classifier(x_train, x_test, y_train, y_test, knn_best_model, knn_params,
                                               "Adult", "KNN")

    # GaussianNB
    nb_best_model, nb_params = classification_cv.GaussianNB(x_train, y_train, fold=4, iterations=20)
    result[constant.GNB] = evaluate_classifier(x_train, x_test, y_train, y_test, nb_best_model, nb_params,
                                               "Adult",
                                               "Gaussian NB")

    # MLP
    mlp_best_model, mlp_params = classification_cv.MLPClassifier(x_train, y_train,
                                                                 hidden_layer_sizes=[(), (10,)],
                                                                 alphas=[0.01, 0.05, 0.5, 1],
                                                                 max_iter=[10, 20, 50, 100],
                                                                 fold=4, iterations=20)
    result[constant.NN] = evaluate_classifier(x_train, x_test, y_train, y_test, mlp_best_model, mlp_params,
                                              "Adult", "MLP")

    export_result(result, "result/classification/adult.json")


def seismic_bumps():
    print("Started training classifiers on seismic bumps data set.")
    x_train, x_test, y_train, y_test = data_processing.seismic_bumps()
    result = {}

    # LR
    lr_best_model, lr_params = classification_cv.logistic_regression(x_train, y_train, C_min=0.01, C_max=10, fold=3,
                                                                     iterations=20)
    result[constant.LR] = evaluate_classifier(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                              "Seismic bumps",
                                              "Logistic Regression")

    # DECISION TREE
    dt_best_model, dt_params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=3, iterations=20)
    result[constant.DECISION_TREE] = evaluate_classifier(x_train, x_test, y_train, y_test, dt_best_model, dt_params,
                                                         "Seismic bumps",
                                                         "DECISION TREE")

    # RANDOM FOREST
    rf_best_model, rf_params = classification_cv.random_forest(x_train, y_train, max_estimator=20, fold=4,
                                                               iterations=20)
    result[constant.RANDOM_FOREST] = evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                                                         "Seismic bumps", "RANDOM FOREST")

    # AdaBoost
    rf_best_model, rf_params = classification_cv.ada_boost_classifier(x_train, y_train, no_estimators=50, fold=4,
                                                                      iterations=20)
    result[constant.ADABOOST] = evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                                                    "Seismic bumps", "AdaBoost")

    # SVC
    svc_best_model, svc_params = classification_cv.SVC(x_train, y_train,
                                                       ['linear', 'poly', 'rbf', 'sigmoid'],
                                                       0.01, 100, 1, 1000, fold=4,
                                                       iterations=20)
    result[constant.SVC] = evaluate_classifier(x_train, x_test, y_train, y_test, svc_best_model, svc_params,
                                               "Seismic bumps", "SVC")

    # KNN
    knn_best_model, knn_params = classification_cv.KNC(x_train, y_train, neighbors=10, fold=4, iterations=20)
    result[constant.KNC] = evaluate_classifier(x_train, x_test, y_train, y_test, knn_best_model, knn_params,
                                               "Seismic bumps", "KNN")

    # GaussianNB
    nb_best_model, nb_params = classification_cv.GaussianNB(x_train, y_train, fold=4, iterations=20)
    result[constant.GNB] = evaluate_classifier(x_train, x_test, y_train, y_test, nb_best_model, nb_params,
                                               "Seismic bumps",
                                               "Gaussian NB")

    # MLP
    mlp_best_model, mlp_params = classification_cv.MLPClassifier(x_train, y_train,
                                                                 hidden_layer_sizes=[(), (10,)],
                                                                 alphas=[0.01, 0.05, 0.5, 1],
                                                                 max_iter=[10, 20, 50, 100],
                                                                 fold=4, iterations=20)
    result[constant.NN] = evaluate_classifier(x_train, x_test, y_train, y_test, mlp_best_model, mlp_params,
                                              "Seismic bumps", "MLP")

    export_result(result, "result/classification/seismic_bumps.json")


def statlog_german():
    print("Started training classifiers on statlog german data set.")
    x_train, x_test, y_train, y_test = data_processing.statlog_german()
    result = {}

    # LR
    lr_best_model, lr_params = classification_cv.logistic_regression(x_train, y_train, C_min=0.01, C_max=10, fold=3,
                                                                     iterations=20)
    result[constant.LR] = evaluate_classifier(x_train, x_test, y_train, y_test, lr_best_model, lr_params,
                                              "Statlog German", "Logistic Regression")

    # DECISION TREE
    dt_best_model, dt_params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=3, iterations=20)
    result[constant.DECISION_TREE] = evaluate_classifier(x_train, x_test, y_train, y_test, dt_best_model, dt_params,
                                                         "Statlog German", "DECISION TREE")

    # RANDOM FOREST
    rf_best_model, rf_params = classification_cv.random_forest(x_train, y_train, max_estimator=20, fold=4,
                                                               iterations=20)
    result[constant.RANDOM_FOREST] = evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                                                         "Statlog German", "RANDOM FOREST")

    # AdaBoost
    rf_best_model, rf_params = classification_cv.ada_boost_classifier(x_train, y_train, no_estimators=50, fold=4,
                                                                      iterations=20)
    result[constant.ADABOOST] = evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                                                    "Statlog German", "AdaBoost")

    # SVC
    svc_best_model, svc_params = classification_cv.SVC(x_train, y_train,
                                                       ['linear', 'poly', 'rbf', 'sigmoid'],
                                                       0.01, 100, 1, 1000, fold=4,
                                                       iterations=20)
    result[constant.SVC] = evaluate_classifier(x_train, x_test, y_train, y_test, svc_best_model, svc_params,
                                               "Statlog German", "SVC")

    # KNN
    knn_best_model, knn_params = classification_cv.KNC(x_train, y_train, neighbors=10, fold=4, iterations=20)
    result[constant.KNC] = evaluate_classifier(x_train, x_test, y_train, y_test, knn_best_model, knn_params,
                                               "Statlog German", "KNN")

    # GaussianNB
    nb_best_model, nb_params = classification_cv.GaussianNB(x_train, y_train, fold=4, iterations=20)
    result[constant.GNB] = evaluate_classifier(x_train, x_test, y_train, y_test, nb_best_model, nb_params,
                                               "Statlog German", "Gaussian NB")

    # MLP
    mlp_best_model, mlp_params = classification_cv.MLPClassifier(x_train, y_train,
                                                                 hidden_layer_sizes=[(), (10,)],
                                                                 alphas=[0.01, 0.05, 0.5, 1],
                                                                 max_iter=[10, 20, 50, 100],
                                                                 fold=4, iterations=20)
    result[constant.NN] = evaluate_classifier(x_train, x_test, y_train, y_test, mlp_best_model, mlp_params,
                                              "Statlog German", "MLP")

    export_result(result, "result/classification/statlog_german.json")


statlog_german()
