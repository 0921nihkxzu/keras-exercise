# optimizers.py 

import numpy as np

class Optimizer:

	pass

class SGD(Optimizer):

	def __init__(self, lr=0.01, momentum=0.0):
		
		self.lr = lr
		self.momentum = momentum
		self.Vdw = 0
		self.Vdb = 0
		self.t = 1

	def update(self, W, b, dW, db):

		lr = self.lr
		b1 = self.momentum
		Vdw = self.Vdw
		Vdb = self.Vdb
		t = self.t

		Vdw = (b1*Vdw + (1-b1)*dW)
		Vdb = (b1*Vdb + (1-b1)*db)

		W = W - lr*Vdw
		b = b - lr*Vdb

		self.Vdw = Vdw
		self.Vdb = Vdb
		self.t += 1

		return W, b

class RMSprop(Optimizer):

	def __init__(self, lr=0.001, rho=0.9, epsilon=1e-9):
		
		self.lr = lr
		self.rho = rho
		self.epsilon = epsilon
		self.Sdw = 0
		self.Sdb = 0
		self.t = 1

	def update(self, W, b, dW, db):

		lr = self.lr
		rho = self.rho
		epsilon = self.epsilon
		Sdw = self.Sdw
		Sdb = self.Sdb
		t = self.t

		Sdw = (rho*Sdw + (1-rho)*dW**2)/(1-rho**t)
		Sdb = (rho*Sdb + (1-rho)*db**2)/(1-rho**t)

		W = W - lr*dW/(np.sqrt(Sdw)+epsilon)
		b = b - lr*db/(np.sqrt(Sdb)+epsilon)

		self.Sdw = Sdw
		self.Sdb = Sdb
		self.t += 1

		return W, b

class Adam(Optimizer):

	def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-9):

		self.lr = lr
		self.epsilon = epsilon
		self.b1 = beta_1
		self.b2 = beta_2

		self.Vdw = 0
		self.Vdb = 0

		self.Sdw = 0
		self.Sdb = 0
		
		self.t = 1

	def update(self, W, b, dW, db):

		lr = self.lr
		epsilon = self.epsilon
		b1 = self.b1
		b2 = self.b2
		Vdw = self.Vdw
		Vdb = self.Vdb
		Sdw = self.Sdw
		Sdb = self.Sdb
		t = self.t

		Vdw = (b1*Vdw + (1-b1)*dW)
		Vdb = (b1*Vdb + (1-b1)*db)

		Sdw = (b2*Sdw + (1-b2)*dW**2)
		Sdb = (b2*Sdb + (1-b2)*db**2)

		W = W - lr*Vdw/(np.sqrt(Sdw)+epsilon)
		b = b - lr*Vdb/(np.sqrt(Sdb)+epsilon)

		self.Vdw = Vdw
		self.Vdb = Vdb
		self.Sdw = Sdw
		self.Sdb = Sdb
		self.t += 1

		return W, b