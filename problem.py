import os

import pandas as pd
import numpy as np


from sklearn.metrics import recall_score
from sklearn.model_selection import ShuffleSplit

import rampwf as rw
from rampwf.score_types.base import BaseScoreType

problem_title = "Bitcoin Ransomeware Detection"

_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
workflow = rw.workflows.Estimator()


class RECALL(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="RECALL", precision=4):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):

        # y_true = np.argmax(y_true, axis=1)
        # y_pred = np.argmax(y_pred, axis=1)

        score = recall_score(y_true=y_true, y_pred=y_pred, average='weighted')

        return score


score_types = [
    # RECALL(name="RECALL"),
    rw.score_types.ROCAUC(name='auc'),
    # rw.score_types.Accuracy(name='acc'),
    rw.score_types.NegativeLogLikelihood(name='nll'),
    # rw.score_types.BrierScore(name='brier_score')
]


_ignore_column_names = ['label']


def _get_data(path=".", split="train"):

    f_name = str(split) + ".csv"
    dataset = pd.read_csv(os.path.join(path, 'data', f_name), sep=" ")

    X = dataset.drop(_ignore_column_names, axis=1)
    y = np.array(np.where(dataset["label"] == "white", 0, 1))

    # X = X_df.to_numpy()
    # y = np.reshape(y_df.to_numpy(), (y_df.shape, ))
    return X, y


def get_train_data(path="."):
    return _get_data(path, "train")


def get_test_data(path="."):
    return _get_data(path, "test")


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)
