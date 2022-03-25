# import os
# import warnings
# import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
from tqdm.notebook import tqdm
# import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from datetime import date, datetime
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pandas import Timestamp
from sklearn.linear_model import LogisticRegression

def _process_time_feature(X):
    
    '''From the columns "year" and "day", we create a datetime column day/month/year'''
    '''For example, for year=2017 and day=130, we will get 10/05/2017'''
    dat_temp = []
    for i in range(X.shape[0]):
        aux = date.fromordinal(date(X['year'][i], 1, 1).toordinal() + X['day'][i]  - 1).strftime('%d/%m/%Y')
        dat_temp.append(aux)
    X['time'] = dat_temp
    X['time'] = pd.to_datetime(X['time'])
    X.pop('day')
    X.pop('year')
    return X['time'].values

def get_estimator():
    
    cnt_featnames = ['length', 'weight', 'count', 'looped', 'neighbors', 'income']
    date_featnames = ['time']
    
    '''First we create a scikit-learn encoder that computes the age in days of columns containing dates'''
    class AgeEncoder(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            self.today = Timestamp.today()
            return self

        def transform(self, X):
            return X.apply(lambda x: (x - self.today).dt.days, axis=0)
        
    standard_scaler = StandardScaler()
        
    '''A pipeline that first computes age, and standardizes it'''
    scaled_age_encoder = Pipeline([
        ('age', AgeEncoder()),
        ('scaling', StandardScaler())
    ])
    
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
