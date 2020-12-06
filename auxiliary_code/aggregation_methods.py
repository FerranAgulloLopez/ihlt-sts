import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


models = {
    'svr': SVR()
}


def mean(config, scores, og_labels, models, test):
    return scores.mean(axis=1)


def svm(config, scores, og_labels, models, test):
    mms_pred = MinMaxScaler()
    mms_features = MinMaxScaler()
    mms_features = mms_features.fit(scores)
    scores = mms_features.transform(scores)
    if not test:
        models['svr'] = models['svr'].fit(scores, og_labels)
    predictions = models['svr'].predict(scores)

    mms_pred = mms_pred.fit(np.reshape(predictions, (len(predictions), 1)))
    predictions = mms_pred.transform(np.reshape(predictions, (len(predictions), 1)))
    return np.reshape(predictions, (np.shape(predictions)[0], ))


def run_aggregation_method(config, scores, og_labels, test=False):
    return eval(config['name'])(config, scores, og_labels, models, test)

