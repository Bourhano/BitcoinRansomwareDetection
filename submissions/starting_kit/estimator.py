import numpy as np
import pandas as pd
from datetime import date

from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import LogisticRegression


def _process_time_feature(X):
    """
    From the columns "year" and "day", we create a datetime
    column day/month/year
    For example, for year=2017 and day=130,
    we will get 10/05/2017
    """
    years = np.array(X['year'])
    days = np.array(X['day'])
    dat_temp = []
    for i in range(X.shape[0]):
        aux = date.fromordinal(
            date(years[i], 1, 1).toordinal()
            + days[i]-1).strftime('%d/%m/%Y')
        dat_temp.append(aux)

    X['time'] = pd.to_datetime(dat_temp)
    return X['time']


def get_estimator():

    cnt_featnames = ['length', 'weight',
                     'count', 'looped', 'neighbors', 'income']
    date_featnames = ['year', 'day']

    '''First we create a scikit-learn encoder that computes the age in
                    days of columns containing dates'''
    class AgeEncoder(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            self.today = pd.Timestamp.today()
            return self

        def transform(self, X):
            aux = (X-self.today).dt.days
            return aux.values.reshape((len(aux), -1))

    '''A pipeline that first computes age, and standardizes it'''
    scaled_age_encoder = FunctionTransformer(
        _process_time_feature, validate=False
    )

    scaled_age_encoder = make_pipeline(
        scaled_age_encoder, AgeEncoder(),
        StandardScaler()
    )

    standard_scaler = StandardScaler()
    '''Let's combine all these transformations'''
    transformer = ColumnTransformer([
        ('standard_scaling', standard_scaler, cnt_featnames),
        ('dates_age_scaled', scaled_age_encoder, date_featnames)
    ])

    '''Now, we define a classifier'''
    classifier = LogisticRegression()

    '''We wrap all in a pipeline'''
    pipeline = Pipeline(steps=[
        ('preprocessing', transformer),
        ('classifier', classifier)
    ])

    return pipeline
