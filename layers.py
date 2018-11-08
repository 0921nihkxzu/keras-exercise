# layers.py
# contains all layer classes

import numpy as np
from meik.utils import activations
from meik.utils import initializations
from meik.initializer import Initializer
from meik.regularizer import Regularizer

class Layer:
	
	def __init__(self):
		
		self.id = None
		self.units = None
		self.inputs = None
		
	def init(self,_id, inputs):
		
		self.id = _id
		self.inputs = inputs
		
class Dense(Layer):
	
	def __init__(self, units = None, activation = None, initialization = None, init_params = None, inputs = None, regularization = None, reg_lambda = 0.01):
		
		Layer.__init__(self)
		
		assert(type(units) == int and units > 0), "Provide positive integer number of units using kwarg 'units'"        
		assert(activation in ['sigmoid','relu','tanh','softmax','linear']), "Provide activation as either 'sigmoid','relu','tanh','softmax' or 'linear' using kwarg 'activation'"
		
		self.id = None
		
		self.inputs = inputs
		self.units = units
		self.activation = activation
		self.learning_rate = None

		# setting initializer
		self.initializer = Initializer(activation = activation, initialization = initialization, init_params = init_params)

		# setting regularizer
		self.regularizer = Regularizer(regularization, reg_lambda)

		# setting activation methods
		self.g = getattr(activations, activation)
		self.dg = getattr(activations, 'd'+activation)

		# declaring important variables            
		self.W = None
		self.b = None
		self.A = None
		self.A0 = None
		
		self.dW = None
		self.db = None

	def init(self, _id, inputs):

		self.id = _id
		self.inputs = inputs
		
		i = self.units
		j = self.inputs
		
		# initialize weights
		self.W = self.initializer.initialize((i,j))

		# initialize biases    
		self.b = np.zeros((i,1))
				
	def forwardprop(self, A0):
		
		W = self.W
		b = self.b
		
		Z = np.matmul(W, A0)+b
		A = self.g(Z)
		
		self.A = A
		self.A0 = A0
		
		return A
		
	def backprop(self, dA0):
		
		A = self.A
		A0 = self.A0
		W = self.W
		
		m = A.shape[1]
		
		dZ = dA0*self.dg(A)
		dW = 1./m*(np.matmul(dZ, A0.T) + self.regularizer.update(W))
		db = 1./m*np.sum(dZ, axis=1, keepdims=True)
		dA = np.matmul(W.T, dZ)
		
		self.dW = dW
		self.db = db
		
		return dA
	
	def backprop_output(self, dZ):
		
		A = self.A
		A0 = self.A0
		W = self.W
		
		m = A.shape[1]
		
		dW = 1./m*(np.matmul(dZ, A0.T) + self.regularizer.update(W))
		db = 1./m*np.sum(dZ, axis=1, keepdims=True)
		dA = np.matmul(W.T, dZ)
		
		self.dW = dW
		self.db = db
		
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