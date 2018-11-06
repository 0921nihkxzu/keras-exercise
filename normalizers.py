# normalizer.py

import numpy as np
from meik.utils import normalizations

class Normalizer:
    
    def __init__(self, method = 'none'):
        
        assert(method in ['none', 'normal', 'bounded']), "Provide method as either 'none', 'normal', 'midrange' using kwarg 'method'"
        self.method = method
        
        self.normalize = getattr(normalizations, self.method)
        self.get_params = getattr(normalizations, self.method+'_params')
        
    def train(self, X):

        assert(np.ndim(X) == 2), "Make sure the input is 2 dimensional e.g. np.shape(X) = (s1, s2) not (s1,)"
        self.params = self.get_params(X)
        
        return self.normalize(X, self.params)
        
    def evaluate(self, X):
        
        return self.normalize(X, self.params)