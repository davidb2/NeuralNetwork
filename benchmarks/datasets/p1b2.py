from __future__ import absolute_import
from ..utils.data_utils import get_file
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score
import gzip
# from ..utils.data_utils import get_file
# from six.moves import cPickle
import sys


seed = 2016
DEMO = ('http://bioseed.mcs.anl.gov/~fangfang/benchmarks/data/P1B2.train.csv', 'http://bioseed.mcs.anl.gov/~fangfang/benchmarks/data/P1B2.test.csv')


def load_data_from_url(shuffle=True, n_cols=None, source=DEMO):
    train_path = get_file('P1B2.train.csv', origin=DEMO[0])
    test_path = get_file('P1B2.test.csv', origin=DEMO[1])

    usecols = list(range(n_cols)) if n_cols else None

    df_train = pd.read_csv(train_path, engine='c', usecols=usecols)
    df_test = pd.read_csv(test_path, engine='c', usecols=usecols)

    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        df_test = df_test.sample(frac=1, random_state=seed)

    X_train = df_train.iloc[:, 2:].as_matrix()
    X_test = df_test.iloc[:, 2:].as_matrix()

    y_train = pd.get_dummies(df_train[['cancer_type']]).as_matrix()
    y_test = pd.get_dummies(df_test[['cancer_type']]).as_matrix()

    return (X_train, y_train), (X_test, y_test)

def load_data_from_file(shuffle=True, n_cols=None, train='', test='', trainvar=''):
    usecols = list(range(n_cols)) if n_cols else None

    df_train = pd.read_csv(train, engine='c', usecols=usecols)
    df_test = pd.read_csv(test, engine='c', usecols=usecols)

    if shuffle:
        df_train = df_train.sample(frac=1, random_state=seed)
        df_test = df_test.sample(frac=1, random_state=seed)

    X_train = df_train.iloc[:, 2:].as_matrix()
    X_test = df_test.iloc[:, 2:].as_matrix()

    y_train = pd.get_dummies(df_train[[trainvar]]).as_matrix()
    y_test = pd.get_dummies(df_test[[trainvar]]).as_matrix()

    return (X_train, y_train), (X_test, y_test)


def evaluate(y_test, y_pred):
    def map_max_indices(nparray):
        maxi = lambda a: a.argmax()
        iter_to_na = lambda i: np.fromiter(i, dtype=np.float)
        return np.array([maxi(a) for a in nparray])
    ya, ypa = tuple(map(map_max_indices, (y_test, y_pred)))
    accuracy = accuracy_score(ya, ypa)
    print('Final accuracy of best model: {}%'.format(100 * accuracy))
    return accuracy