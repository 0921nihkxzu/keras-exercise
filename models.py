# models.py
# Contains the network model class

import numpy as np
from meik.utils.losses import *
from meik.layers import Layer
from meik.metrics import Metrics
from meik.normalizers import Normalizer
from meik.layers import Dropout
from meik import optimizers
import copy

class Sequential:
	
	def __init__(self):
		
		self.layers = []
		self.params = {
			'learning_rate': None,
			'epochs': None,
			'loss': None,
		}
		
		self.batch_metrics = []
		self.epoch_metrics = []
		
	def add(self,layer):
		
		assert (issubclass(type(layer), Layer))
		
		_id = len(self.layers)
		if _id == 0:
			assert(type(layer.inputs) == int), "Provide number of inputs for initial layer"
			inputs = layer.inputs
		else:
			inputs = self.layers[-1].units
			
		layer.init(_id, inputs)
		self.layers.append(layer)
			
	def build(self, loss = None, normalization = 'none', optimizer = 'SGD', train_metrics = None, eval_metrics = None, thresholds = np.array([0.5])):
		
		self.params['loss'] = loss
		self.params['normalization'] = normalization
		self.params['optimizer'] = optimizer
		
		self.normalize = Normalizer(method = normalization)
		
		self.metrics = Metrics(loss = loss, train_metrics = train_metrics, eval_metrics = eval_metrics, thresholds = thresholds)

		if type(optimizer) == str:
			assert(optimizer in ['SGD', 'RMSprop', 'Adam']), "If providing optimizer as a string provide string as SGD', 'RMSprop' or 'Adam' -- note: default parameters will be used"
			self.optimizer = getattr(optimizers, optimizer)()
		else:
			assert(issubclass(type(optimizer), optimizers.Optimizer)), "Provide optimizer as string or optimizer object"
			self.optimizer = optimizer

		for i in range(len(self.layers)):
			self.layers[i].optimizer = copy.deepcopy(self.optimizer)

	def predict(self,X):
		
		layers = self.layers
		
		A = X
		for i in range(len(layers)):
			A = layers[i].predict(A)
		
		return A

	def grad_check_predict(self,X):
		
		layers = self.layers
		
		A = X
		for i in range(len(layers)):
			A = layers[i].grad_check_predict(A)
		
		return A

	def forwardprop(self, X):

		layers = self.layers
		
		A = X
		for i in range(len(layers)):
			A = layers[i].forwardprop(A)
		
		return A

	def backprop(self, Y, A):
		
		layers = self.layers
		
		if self.params["loss"] in ['mse', 'binary_crossentropy', 'categorical_crossentropy']:
			dZ = A - Y
		elif self.params["loss"] == 'mae':
			dZ = np.sign(A - Y)
		dA = self.layers[-1].backprop_output(dZ)
		for i in range(len(layers)-2,-1,-1):
			dA = layers[i].backprop(dA)
	
	def update(self):

		layers = self.layers

		for i in range(len(layers)-1,-1,-1):
			layers[i].update()

	def regularization_loss(self, m):

		loss = 0.
		for l in self.layers:
			try:
				loss += l.regularizer.loss(l.W, m)
			except AttributeError:
				continue

		return loss
			
	def train(self, X, Y, batch_size=128, epochs=1, verbose=1):

		m = Y.shape[1]
		
		layers = self.layers
		
		X_norm = self.normalize.train(X)

		assert(batch_size <= m), "Batch size must be less than or equal to training examples m"
		batches = int(m/batch_size-1e-9)

		for i in range(epochs):
			
			A = np.zeros(Y.shape)

			for j in range(batches+1):

				Xt = X_norm[:, j*batch_size:(j+1)*batch_size]
				Yt = Y[:, j*batch_size:(j+1)*batch_size]

				At = self.forwardprop(Xt)
				self.backprop(Yt, At)
				self.update()
			
				reg_loss = self.regularization_loss(batch_size)
				batch_metrics = self.metrics.train(Yt, At, reg_loss)
				self.batch_metrics.append(batch_metrics)

				A[:, j*batch_size:(j+1)*batch_size] = At
			
			reg_loss = self.regularization_loss(m)
			epoch_metrics = self.metrics.train(Y, A, reg_loss)
			self.epoch_metrics.append(epoch_metrics)

			if verbose == 1:
				self.metrics.train_print(i, epochs)
			
		print("------------ Final performance ------------\n")
		self.metrics.train_print(i, epochs)
		
	def evaluate(self, X, Y):

		m = Y.shape[1]
		
		X = self.normalize.evaluate(X)
			
		A = self.predict(X)
		
		reg_loss = self.regularization_loss(m)
		score = self.metrics.evaluate(Y, A, reg_loss)
		
		self.score = score
		
		return score
		