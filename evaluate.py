# Creator: Hoang-Dung Do

import sklearn.metrics
import classification_cv
import data_processing


def diabete_retinopathy():
    print("Started training classifiers on diabete retinopathy data set.")
    x_train, x_test, y_train, y_test = data_processing.diabetic_retinopathy()

    best_model, params = classification_cv.decision_trees(x_train, y_train, max_depth=10, fold=4)
    y_predict = best_model.predict(x_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_predict)
    print("--Decision tree:")
    print("\tDiabetic retinopathy accuracy: {0:.2f}%".format(accuracy * 100))
    print("\tHyperparam max_depth: {0}".format(params['max_depth']))

diabete_retinopathy()
