# standardization.py
# standardization functions

import numpy as np

def normal(X):
    
    # Returns the standardized matrix X_norm with with each row having mean = 0 and a std = 1

    assert(np.ndim(X) == 2), "Make sure the input is 2 dimensional e.g. np.shape(X) = (s1, s2) not (s1,)"
    
    mu = np.mean(X,axis=1).reshape(X.shape[0],1)
    std = np.std(X,axis=1).reshape(X.shape[0],1)
    
    X_norm = (X-mu)/std
    
    return X_norm, mu, std

def midrange(X):
    
    # Returns the standardized matrix X_norm with each row having a range of [-1., 1]
    
    assert(np.ndim(X) == 2), "Make sure the input is 2 dimensional e.g. np.shape(X) = (s1, s2) not (s1,)"
    
    max_ = np.amax(X, axis = 1).reshape(X.shape[0],1)
    min_ = np.amin(X, axis = 1).reshape(X.shape[0],1)
    
    midrange = (max_ + min_)/2
    range_ = max_ - min_
    
    X_norm = 2*(X-midrange)/range_
    
    return X_norm, midrange, range_