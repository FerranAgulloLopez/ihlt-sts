import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge


models = {
    'svr': GridSearchCV(SVR(kernel='rbf', gamma=0.1), param_grid={"C": [0.1, 1, 10, 100, 1000],
                                                                "gamma": np.logspace(-2, 2, 5)}),
    'krr': GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1), param_grid={"alpha": [10, 1e0, 0.1, 1e-2, 1e-3],
                                                                          "gamma": np.logspace(-2, 2, 5)})
}


def mean(config, scores, og_labels, models, test):
    return scores.mean(axis=1)


def svm(config, scores, og_labels, models, test):
    mms_pred = MinMaxScaler(feature_range=(0, 5))
    mms_features = MinMaxScaler()

    # Standarize features
    mms_features = mms_features.fit(scores)
    scores = mms_features.transform(scores)

    # Train or predict with SVR
    if not test:
        models['svr'] = models['svr'].fit(scores, og_labels)
        print(models['svr'].best_params_)
    predictions = models['svr'].predict(scores)

    predictions = mms_pred.fit_transform(np.reshape(predictions, (len(predictions), 1)))

    return np.reshape(predictions, (np.shape(predictions)[0], ))


def krr(config, scores, og_labels, models, test):
    mms_pred = MinMaxScaler(feature_range=(0, 5))
    mms_features = MinMaxScaler()

    # Standarize features
    mms_features = mms_features.fit(scores)
    scores = mms_features.transform(scores)

    # Train or predict with SVR
    if not test:
        models['krr'] = models['krr'].fit(scores, og_labels)
        print(models['krr'].best_params_)
    predictions = models['krr'].predict(scores)

    predictions = mms_pred.fit_transform(np.reshape(predictions, (len(predictions), 1)))

    return np.reshape(predictions, (np.shape(predictions)[0], ))


def run_aggregation_method(config, scores, og_labels, test=False):
    return eval(config['name'])(config, scores, og_labels, models, test)

