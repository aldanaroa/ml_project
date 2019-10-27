# Creator: Hoang-Dung Do

import numpy as np
import sklearn.tree
import sklearn.model_selection
import scipy.stats


def decision_trees(x_train, y_train, max_depth=10, fold=5):
    dtc = sklearn.tree.DecisionTreeClassifier(random_state=0)
    params = {
        "max_depth": range(1, max_depth + 1)
    }

    random_search_cv = sklearn.model_selection.RandomizedSearchCV(dtc, param_distributions=params, verbose=1, cv=fold,
                                                                  random_state=0, n_iter=20)
    print("Training decision tree classifier ...")
    random_search_cv.fit(x_train, y_train)

    return random_search_cv.best_estimator_, random_search_cv.best_params_
