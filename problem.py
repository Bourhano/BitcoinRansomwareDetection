import os
from glob import glob
import pandas as pd
from sklearn.model_selection import ShuffleSplit

import rampwf as rw

problem_title = "Bitcoin Ransomware Detection"

_prediction_label_names = [-1, 1]

Predictions = rw.prediction_types.make_multiclass(
    label_names=_prediction_label_names)
workflow = rw.workflows.Estimator()


score_types = [
    rw.score_types.ROCAUC(name='auc')
]


_ignore_column_names = 'address'
_target_column_name = 'label'


def get_file_list_from_dir(*, path, filename):
    data_files = sorted(glob(os.path.join(path, "data/public", filename)))
    return data_files


def _get_data(path, f_name):
    data_files = get_file_list_from_dir(path=path, filename=f_name)
    dataset = pd.concat((pd.read_csv(f) for f in data_files))

    X = dataset.drop([_target_column_name] + [_ignore_column_names], axis=1)
    y = dataset[_target_column_name].values

    return X, y


def get_train_data(path="."):
    f_name = "train.csv.gz"
    return _get_data(path, f_name)


def get_test_data(path="."):
    f_name = "test.csv.gz"
    return _get_data(path, f_name)


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=10, test_size=0.25, random_state=57)
    return cv.split(X, y)
