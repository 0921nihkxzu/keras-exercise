# regularizer.py

import numpy as np

class Regularizer:

	def __init__(self, regularization = None, lambda_ = 0.01):

		assert(regularization in [None, 'l1', 'l2']), "Provide regularization type as 'l1' or 'l2'"

		self.regularization = regularization
		self.lambda_ = lambda_

		if regularization == None:
			self.loss = lambda W: 0
			self.update = lambda W: 0
		else:
			assert(type(lambda_) == np.float64 or type(lambda_) == float), "Provide lambda as a float"
			self.loss = getattr(self, regularization)
			self.update = getattr(self, regularization+'_update')


	def l2(self, W):
		return np.sum(W**2)*self.lambda_/2.

	def l2_update(self, W):
		return W*self.lambda_

	def l1(self, W):
		return np.sum(abs(W))*self.lambda_

	def l1_update(self, W):
		return np.sign(W)*self.lambda_

