# normalizer.py

import numpy as np
from meik.utils.normalizations import *

class Normalizer:
    
    def __init__(self, method = 'none'):
        
        assert(method in ['none', 'normal', 'bounded']), "Provide method as either 'none', 'normal', 'midrange' using kwarg 'method'"
        self.method = method
        
        # Methods for each type of standardization
        # 1. Empty dummy for no normalization
        
        self.none = lambda X, params: X
        self.none_params = lambda X: [None, None]
        
        # 2. With mean and standard deviation
        self.normal = normal
        self.normal_params = normal_params
        
        # 3. With midrange and range
        self.bounded = bounded
        self.bounded_params = bounded_params
        
        # Setting standardization method
        self.normalize = getattr(self, self.method)
        self.get_params = getattr(self, self.method+'_params')
        
    def train(self, X):

        assert(np.ndim(X) == 2), "Make sure the input is 2 dimensional e.g. np.shape(X) = (s1, s2) not (s1,)"
        self.params = self.get_params(X)
        
        return self.normalize(X, self.params)
        
    def evaluate(self, X):
        
        return self.normalize(X, self.params)