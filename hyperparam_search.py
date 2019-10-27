# Creator: Hoang-Dung Do
# This source file contains functions to search for optimal hyperparameters using cross validation with random search
import numpy as np
import sklearn
import sklearn.model_selection


def best_model(model_, hyper_params_, training_data_, test_data_, fold_=5, iter_=20):
    """
    This function evaluate trained models with given a data set in a given model family with different hyperparameters
    :param model_: model family
    :param hyper_params_: hyperparameters to be evaluated
    :param training_data_: a training data set
    :param test_data_: a testing data set
    :param fold_: number of fold in cross validation
    :param iter_: number of iteration
    :return: the parameters corresponding to the best model chosen.
    """

    