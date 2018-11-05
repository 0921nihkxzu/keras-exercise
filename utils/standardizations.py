# standardizations.py

import numpy as np

# Standardizes features with mean = 0, standard deviation = 1

def normal(X, params):

    mu, std = params

    return (X-mu)/std

def normal_params(X):

    mu = np.mean(X,axis=1).reshape(X.shape[0],1)
    std = np.std(X,axis=1).reshape(X.shape[0],1)

    return [mu,std]

# Standardizes features within range [-1, 1]

def bounded(X, params):

    midrange, range_ = params

    return 2*(X-midrange)/range_

def bounded_params(X):

    max_ = np.amax(X, axis = 1).reshape(X.shape[0],1)
    min_ = np.amin(X, axis = 1).reshape(X.shape[0],1)

    midrange = (max_ + min_)/2
    range_ = max_ - min_

    return [midrange, range_]