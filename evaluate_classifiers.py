# Creator: Hoang-Dung Do

import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import pandas as pd
import classification_cv
import regression_cv
import data_processing


def evaluate_classifier(x_train, x_test, y_train, y_test, model, params, dataset_name, model_name):
    train_accuracy = model.score(x_train, y_train)
    test_accuracy = model.score(x_test, y_test)
    print("--{0}:".format(model_name))
    print("\tTraining accuracy: {1:.2f}%".format(dataset_name, train_accuracy * 100))
    print("\tTesting accuracy: {1:.2f}%".format(dataset_name, test_accuracy * 100))
    if params is not None and len(params.keys()) > 0:
        print("\tHyperparam:")
        for hyperparam in params.keys():
            print("\t\t {0}: {1}".format(hyperparam, params[hyperparam]))
    return train_accuracy, test_accuracy


def diabete_retinopathy():
    print("Started training classifiers on diabete retinopathy data set.")
    x_train, x_test, y_train, y_test = data_processing.diabetic_retinopathy()

    # LR
    lr_best_model, lr_params = classification_cv.logistic_regression(x_train, y_train, C_min=0.01, C_max=10, fold=3,
                                                                     iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, lr_best_model, lr_params, "Diabete retinopathy",
                        "Logistic Regression")

    # DECISION TREE
    dt_best_model, dt_params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=3, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, dt_best_model, dt_params, "Diabete retinopathy",
                        "DECISION TREE")

    # RANDOM FOREST
    rf_best_model, rf_params = classification_cv.random_forest(x_train, y_train, max_estimator=20, fold=4,
                                                               iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                        "Diabete retinopathy", "RANDOM FOREST")

    # SVC
    svc_best_model, svc_params = classification_cv.SVC(x_train, y_train,
                                                       ['linear', 'poly', 'rbf', 'sigmoid'],
                                                       0.01, 100, 1, 1000, fold=4,
                                                       iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, svc_best_model, svc_params, "Diabete retinopathy", "SVC")

    # KNN
    knn_best_model, knn_params = classification_cv.KNC(x_train, y_train, neighbors=10, fold=4, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, knn_best_model, knn_params, "Diabete retinopathy", "KNN")

    # GaussianNB
    nb_best_model, nb_params = classification_cv.GaussianNB(x_train, y_train, fold=4, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, nb_best_model, nb_params, "Diabete retinopathy",
                        "Gaussian NB")

    # MLP
    mlp_best_model, mlp_params = classification_cv.MLPClassifier(x_train, y_train,
                                                                 hidden_layer_sizes=[(), (5,)],
                                                                 alphas=[0.01, 0.05, 0.5, 1],
                                                                 max_iter=[10, 20, 50, 100],
                                                                 fold=4, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, mlp_best_model, mlp_params,
                        "Diabete retinopathy", "MLP")


def default_credit_card():
    print("Started training classifiers on default credit card clients data set.")
    x_train, x_test, y_train, y_test = data_processing.credit_card_client()

    # LR
    lr_best_model, lr_params = classification_cv.logistic_regression(x_train, y_train, C_min=0.01, C_max=10, fold=3,
                                                                     iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, lr_best_model, lr_params, "Default credit card",
                        "Logistic Regression")

    # DECISION TREE
    dt_best_model, dt_params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=3, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, dt_best_model, dt_params, "Default credit card",
                        "DECISION TREE")

    # RANDOM FOREST
    rf_best_model, rf_params = classification_cv.random_forest(x_train, y_train, max_estimator=20, fold=4,
                                                               iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                        "Default credit card", "RANDOM FOREST")

    # SVC
    svc_best_model, svc_params = classification_cv.SVC(x_train, y_train,
                                                       ['linear', 'poly', 'rbf', 'sigmoid'],
                                                       0.01, 100, 1, 1000, fold=3,
                                                       iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, svc_best_model, svc_params, "Default credit card", "SVC")

    # KNN
    knn_best_model, knn_params = classification_cv.KNC(x_train, y_train, neighbors=10, fold=3, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, knn_best_model, knn_params, "Default credit card", "KNN")

    # GaussianNB
    nb_best_model, nb_params = classification_cv.GaussianNB(x_train, y_train, fold=4, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, nb_best_model, nb_params, "Default credit card",
                        "Gaussian NB")

    # MLP
    mlp_best_model, mlp_params = classification_cv.MLPClassifier(x_train, y_train,
                                                                 hidden_layer_sizes=[(), (10,)],
                                                                 alphas=[0.01, 0.05, 0.5, 1],
                                                                 max_iter=[10, 20, 50, 100],
                                                                 fold=4, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, mlp_best_model, mlp_params,
                        "Breast cancer", "MLP")


def breast_cancer():
    print("Started training classifiers on breast cancer data set.")
    x_train, x_test, y_train, y_test = data_processing.breast_cancer()

    # DECISION TREE
    dt_best_model, dt_params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=3, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, dt_best_model, dt_params, "Breast cancer",
                        "DECISION TREE")

    # RANDOM FOREST
    rf_best_model, rf_params = classification_cv.random_forest(x_train, y_train, max_estimator=20, fold=4,
                                                               iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                        "Breast cancer", "RANDOM FOREST")

    # SVC
    svc_best_model, svc_params = classification_cv.SVC(x_train, y_train,
                                                       ['linear', 'poly', 'rbf', 'sigmoid'],
                                                       0.01, 100, 1, 1000, fold=4,
                                                       iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, svc_best_model, svc_params, "Breast cancer", "SVC")

    # KNN
    knn_best_model, knn_params = classification_cv.KNC(x_train, y_train, neighbors=10, fold=4, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, knn_best_model, knn_params, "Breast cancer", "KNN")

    # GaussianNB
    nb_best_model, nb_params = classification_cv.GaussianNB(x_train, y_train, fold=4, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, nb_best_model, nb_params, "Breast cancer",
                        "Gaussian NB")

    # MLP
    mlp_best_model, mlp_params = classification_cv.MLPClassifier(x_train, y_train,
                                                                 hidden_layer_sizes=[(), (10,)],
                                                                 alphas=[0.01, 0.05, 0.5, 1],
                                                                 max_iter=[10, 20, 50, 100],
                                                                 fold=4, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, mlp_best_model, mlp_params,
                        "Breast cancer", "MLP")

def adult():
    print("Started training classifiers on Adult data set.")
    x_train, x_test, y_train, y_test = data_processing.adult()

    # LR
    lr_best_model, lr_params = classification_cv.logistic_regression(x_train, y_train, C_min=0.01, C_max=10, fold=3,
                                                                         iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, lr_best_model, lr_params, "Adult",
                            "Logistic Regression")



    # DECISION TREE
    dt_best_model, dt_params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=3, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, dt_best_model, dt_params, "Adult",
                        "DECISION TREE")

    # RANDOM FOREST
    rf_best_model, rf_params = classification_cv.random_forest(x_train, y_train, max_estimator=20, fold=4,
                                                               iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, rf_best_model, rf_params,
                        "Adult", "RANDOM FOREST")

    # SVC
    #svc_best_model, svc_params = classification_cv.SVC(x_train, y_train,
                                                       ['linear', 'poly', 'rbf', 'sigmoid'],
                                                       0.01, 100, 1, 1000, fold=4,
                                                       iterations=20)
    #evaluate_classifier(x_train, x_test, y_train, y_test, svc_best_model, svc_params, "Adult", "SVC")

    # KNN
    knn_best_model, knn_params = classification_cv.KNC(x_train, y_train, neighbors=10, fold=4, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, knn_best_model, knn_params, "Adult", "KNN")

    # GaussianNB
    nb_best_model, nb_params = classification_cv.GaussianNB(x_train, y_train, fold=4, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, nb_best_model, nb_params, "Adult",
                        "Gaussian NB")

    # MLP
    mlp_best_model, mlp_params = classification_cv.MLPClassifier(x_train, y_train,
                                                                 hidden_layer_sizes=[(), (10,)],
                                                                 alphas=[0.01, 0.05, 0.5, 1],
                                                                 max_iter=[10, 20, 50, 100],
                                                                 fold=4, iterations=20)
    evaluate_classifier(x_train, x_test, y_train, y_test, mlp_best_model, mlp_params,
                        "Adult", "MLP")
