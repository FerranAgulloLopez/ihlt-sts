import numpy as np


def mean(config, scores):
    return scores.mean(axis=1)


def run_aggregation_method(config, scores):
    return eval(config['name'])(config, scores)

