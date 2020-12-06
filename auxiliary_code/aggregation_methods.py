import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR


models = {
    'svr': SVR()
}


def mean(config, scores, og_labels, models, test):
    return scores.mean(axis=1)


def svm(config, scores, og_labels, models, test):
    mms = MinMaxScaler()
    if not test:
        models['svr'] = models['svr'].fit(scores, og_labels)
    predictions = models['svr'].predict(scores)

    mms = mms.fit(np.reshape(predictions, (len(predictions), 1)))
    predictions = mms.transform(np.reshape(predictions, (len(predictions), 1)))
    return np.reshape(predictions, (np.shape(predictions)[0], ))


def run_aggregation_method(config, scores, og_labels, test=False):
    return eval(config['name'])(config, scores, og_labels, models, test)

