# initializer.py

from meik.utils import initializations

class Initializer:
	
	def __init__(self, activation = None, initialization = None, init_params = None):

		assert(activation in ['sigmoid','relu','tanh','softmax','linear']), "Provide activation as either 'sigmoid','relu','tanh','softmax' or 'linear' using kwarg 'activation'"
		assert(initialization in ['default', 'normal', 'uniform', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform']), "Provide initialization as either 'normal', 'uniform', 'truncated_normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal', 'lecun_uniform'"

		self.activation = activation

		# default initializations dependent upon activation
		default_init = {
			'sigmoid': 'glorot_uniform',
			'relu': 'he_uniform',
			'tanh': 'glorot_uniform',
			'softmax': 'glorot_uniform',
			'linear': 'glorot_uniform'
		}

		if initialization == 'default':
			initialization = default_init[activation]

		self.initialization = initialization
		self.init_params = init_params

	def initialize(self, shape):
		
		# setting initialization method
		self.initialize_ = getattr(initializations, self.initialization)

		# initializations requiring parameters
		if self.initialization in ['normal', 'uniform', 'truncated_normal'] and self.init_params != None:
			return self.initialize_(shape, params = self.init_params)
		else:
			return self.initialize_(shape)