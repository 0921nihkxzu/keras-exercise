# layers.py
# contains all layer classes

import numpy as np
from meik.utils.activations import *

class Layer:
    
    def __init__(self):
        
        self.id = None
        self.units = None
        self.inputs = None
        
    def init(self,_id, inputs):
        
        self.id = _id
        self.inputs = inputs
        
class Dense(Layer):
    
    def __init__(self, units = None, activation = None, inputs = None):
        
        Layer.__init__(self)
        
        assert(type(units) == int and units > 0), "Provide positive integer number of units using kwarg 'units'"        
        assert(activation in ['sigmoid','relu','tanh','softmax','linear']), "Provide activation as either 'sigmoid','relu','tanh','softmax' or 'linear' using kwarg 'activation'"
        
        self.id = None
        
        self.inputs = inputs
        self.units = units
        self.activation = activation
        self.learning_rate = None
        
        # setting activation
        if activation == 'sigmoid':
            self.g = sigmoid
            self.dg = dsigmoid
        elif activation == 'relu':
            self.g = relu
            self.dg = drelu
        elif activation == 'tanh':
            self.g = tanh
            self.dg = dtanh
        elif activation == 'softmax':
            self.g = softmax
            self.dg = None #not used since only output layer
        elif activation == 'linear':
            self.g = linear
            self.dg = dlinear
            
        self.Z = None
        self.W = None
        self.b = None
        self.A = None
        self.A0 = None
        
        self.dZ = None
        self.dW = None
        self.db = None
        self.dA = None

    def init(self, _id, inputs):

        self.id = _id
        self.inputs = inputs
        
        i = self.units
        j = self.inputs
        
        # initialization factor
        activation = self.activation
        if activation in ['sigmoid', 'tanh', 'softmax']:
            c = np.sqrt(1/j)
        elif activation in ['relu', 'linear']:
            c = np.sqrt(2/j)
            
        # random initialization from normal distribution mean = 0, std = 1
        self.W = np.random.randn(i,j)*c
        self.b = np.zeros((i,1))
                
    def forwardprop(self, A0):
        
        W = self.W
        b = self.b
        
        Z = np.matmul(W, A0)+b
        A = self.g(Z)
        
        self.Z = Z
        self.A = A
        self.A0 = A0
        
        return A
        
    def backprop(self, dA0):
        
        Z = self.Z
        A = self.A
        A0 = self.A0
        W = self.W
        
        m = Z.shape[1]
        
        dZ = dA0*self.dg(A)
        dW = 1./m*np.matmul(dZ, A0.T)
        db = 1./m*np.sum(dZ, axis=1, keepdims=True)
        dA = np.matmul(W.T, dZ)
        
        self.dZ = dZ
        self.dW = dW
        self.db = db
        self.dA = dA
        
        return dA
    
    def backprop_output(self, dZ):
        
        A = self.A
        A0 = self.A0
        W = self.W
        
        m = A.shape[1]
        
        dW = 1./m*np.matmul(dZ, A0.T)
        db = 1./m*np.sum(dZ, axis=1, keepdims=True)
        dA = np.matmul(W.T, dZ)
        
        self.dZ = dZ
        self.dW = dW
        self.db = db
        self.dA = dA
        
        return dA
    
    def update(self):
        
        a = self.learning_rate
        W = self.W
        b = self.b
        dW = self.dW
        db = self.db
        
        W = W - a*dW
        b = b - a*db
        
        self.W = W
        self.b = b