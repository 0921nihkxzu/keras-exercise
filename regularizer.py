# regularizer.py

import numpy as np

class Regularizer:

	def __init__(self, regularization = None, lambda_ = 0.01):

		assert(regularization in [None, 'l1', 'l2']), "Provide regularization type as 'l1' or 'l2'"
		assert(type(lambda_) == np.float64 or type(lambda_) == float), "Provide lambda as a float"

		self.regularization = regularization
		self.lambda_ = lambda_

		if regularization == None:
			self.loss = lambda W, m: 0
			self.update = lambda W, m: 0
		else:
			self.loss = getattr(self, regularization)
			self.update = getattr(self, regularization+'_update')


	def l2(self, W, m):
		c = self.lambda_/(2.*m)
		return c*np.sum(W**2)

	def l2_update(self, W, m):
		c = self.lambda_/m
		return c*W

	def l1(self, W, m):
		c = self.lambda_/m
		return c*np.sum(abs(W))

	def l1_update(self, W, m):
		c = self.lambda_/m
		return c*np.sign(W)

