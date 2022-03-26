import os

import pandas as pd

from sklearn.metrics import recall_score
from sklearn.model_selection import ShuffleSplit

import rampwf as rw

problem_title = "Bitcoin Ransomeware Detection"

_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
workflow = rw.workflows.Estimator()


score_types = [
    rw.score_types.ROCAUC(name='auc')
]


_ignore_column_names = 'address'
_target_column_name = 'label'


def _get_data(path, f_name):

    dataset = pd.read_csv(os.path.join(path, 'data', f_name), sep=",")

    X = dataset.drop([_target_column_name] + [_ignore_column_names], axis=1)
    y = dataset[_target_column_name].values

    return X, y


def get_train_data(path="."):
    f_name = "train.csv"
    return _get_data(path, f_name)


def get_test_data(path="."):
    f_name = "test.csv"
    return _get_data(path, f_name)


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)
