# Creator: Hoang-Dung Do

import cifar_dataprocess
import sklearn.tree
import sklearn.metrics


def evaluate():
    x_train, x_test, y_train, y_test = cifar_dataprocess.load_data()

    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    dtc = sklearn.tree.DecisionTreeClassifier(random_state=0)
    params = {
        "max_depth": range(1, 100)
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(dtc, param_distributions=params, verbose=1,
                                                                  cv=5, random_state=0, n_iter=20)
    print("\nTraing on CIFAR10 data...")
    random_search_cv.fit(x_train, y_train)

    y_pred = random_search_cv.best_estimator_.predict(x_test)
    accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
    print("\tAccuracy: %.2f%%" % (accuracy * 100))

    best_params = random_search_cv.best_params_
    if best_params is not None and len(params.keys()) > 0:
        print("\tHyperparam:")
        for hyperparam in best_params.keys():
            print("\t\t {0}: {1}".format(hyperparam, best_params[hyperparam]))
