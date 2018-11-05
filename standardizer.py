# standardizer.py

import numpy as np
from meik.utils.standardizations import *

class Standardizer:
    
    def __init__(self, method = 'none'):
        
        assert(method in ['none', 'normal', 'bounded']), "Provide method as either 'none', 'normal', 'midrange' using kwarg 'method'"
        self.method = method
        self.get_params = getattr(self, self.method+'_params')
        self.normalize = getattr(self, self.method)
        
    def train(self, X):

        assert(np.ndim(X) == 2), "Make sure the input is 2 dimensional e.g. np.shape(X) = (s1, s2) not (s1,)"
        self.params = self.get_params(X)
        
        return self.normalize(X, self.params)
        
    def evaluate(self, X):
        
        return self.normalize(X, self.params)
    
    # Methods for each type of standardization
    # 1. Empty dummy for no normalization
    
    def none(self, X, params):

        return X

    def none_params(self, X):

        return [None,None]
    
    # 2. With mean and standard deviation
    
    def normal(self, X, params):
        return normal(X, params)

    def normal_params(self, X):
        return normal_params(X)

    # 3. With midrange and range
    
    def bounded(self, X, params):
        return bounded(X, params)

    def bounded_params(self, X):
        return bounded_params(X)