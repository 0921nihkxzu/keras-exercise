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

	def forwardprop(self, A0):
		pass

	def backprop(self, dA0):
		pass

	def update(self):
		pass

	def predict(self, A0):
		pass
		
class Dense(Layer):
	
	def __init__(self, units = None, activation = None, initialization = None, init_params = None, inputs = None, regularization = None, reg_lambda = 0.01):
		
		Layer.__init__(self)
		
		assert(type(units) == int and units > 0), "Provide positive integer number of units using kwarg 'units'"        
		assert(activation in ['sigmoid','relu','tanh','softmax','linear']), "Provide activation as either 'sigmoid','relu','tanh','softmax' or 'linear' using kwarg 'activation'"
		
		self.id = None
		
		self.inputs = inputs
		self.units = units
		self.activation = activation
		self.optimizer = None #this gets assigned by model in model.build()

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

		# setting prediction of layer as forwardprop
		self.predict = self.forwardprop
		self.grad_check_predict = self.forwardprop

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
		dW = 1./m*(np.matmul(dZ, A0.T)) + self.regularizer.update(W, m)
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
		
		dW = 1./m*(np.matmul(dZ, A0.T)) + self.regularizer.update(W, m)
		db = 1./m*np.sum(dZ, axis=1, keepdims=True)
		dA = np.matmul(W.T, dZ)
		
		self.dW = dW
		self.db = db
		
		return dA
	
	def update(self):
		
		W = self.W
		b = self.b
		dW = self.dW
		db = self.db
		
		W, b = self.optimizer.update(W, b, dW, db)
		
		self.W = W
		self.b = b

class Dropout(Layer):

	def __init__(self, keep_prob = 0.9):

		self.keep_prob = keep_prob
		self.grad_check_predict = self.predict

	def init(self,_id, inputs):
		
		self.id = _id
		self.inputs = inputs
		self.units = inputs

	def forwardprop(self,A):
		
		p = self.keep_prob

		shape = (A.shape[0],1)
		mask = np.random.random(shape) < p

		A *= mask*(1/p)

		self.mask = mask

		return A

	def backprop(self,dA):

		p = self.keep_prob
		mask = self.mask

		dA *= mask*(1/p)

		return dA

	def predict(self,A):
		return A

class Batch_norm(Layer):

	def __init__(self, exp_w = 0.90, epsilon = 1e-9):

		self.epoch_mean = 0.
		self.epoch_var = 0.
		
		self.epsilon = epsilon
		self.a = exp_w

		self.gamma = None
		self.beta = None

		self.optimizer = None #this gets assigned by model in model.build()

		self.grad_check_predict = self.forwardprop

	def init(self,_id, inputs):
		
		self.id = _id
		self.inputs = inputs
		self.units = inputs

		self.gamma = np.ones((inputs,1))
		self.beta = np.zeros((inputs,1))

	def forwardprop(self, Z):

		eps = self.epsilon
		a = self.a

		g = self.gamma
		b = self.beta

		u = np.mean(Z, axis = 1, keepdims = True)
		var = np.var(Z, axis = 1, keepdims = True)

		Z_norm = (Z - u)/np.sqrt(var + eps) 

		Z_t = g*Z_norm + b

		self.Z = Z
		self.Z_norm = Z_norm
		self.u = u
		self.var = var

		self.epoch_var = (a*self.epoch_var+(1-a)*var)
		self.epoch_mean = (a*self.epoch_mean+(1-a)*u)

		return Z_t

	def backprop(self, dZ_t):

		Z_norm = self.Z_norm
		g = self.gamma

		var = self.var
		eps = self.epsilon

		Z = self.Z
		u = self.u

		m = Z.shape[1]

		dZ_norm = dZ_t*g

		dvar = np.sum(dZ_norm * (Z - u), axis = 1, keepdims=True) * -1./2*(var+eps)**(-3./2)

		du = np.sum(dZ_norm * -1./np.sqrt(var+eps), axis = 1, keepdims=True) + dvar * -2./m*np.sum(Z - u, axis = 1, keepdims=True)

		dZ = dZ_norm*1./np.sqrt(var+eps) + dvar*2./m*(Z - u) + 1./m*du

		dg = np.sum(dZ_t*Z_norm, axis = 1, keepdims=True)

		db = np.sum(dZ_t, axis = 1, keepdims=True)

		self.dg = dg
		self.db = db

		return dZ

	def predict(self, Z):

		var = self.epoch_var
		u = self.epoch_mean

		eps = self.epsilon

		g = self.gamma
		b = self.beta

		Z_norm = (Z - u)/np.sqrt(var + eps) 

		Z_t = g*Z_norm + b

		return Z_t

	def update(self):

		g = self.gamma
		b = self.beta
		dg = self.dg
		db = self.db

		g, b = self.optimizer.update(g, b, dg, db)

		self.gamma = g
		self.beta = b

from meik.layers import Layer
from meik.initializer import Initializer
from meik.utils import activations
from meik.utils.convolution import im2col, im2col_idxs, padding, depadding

class Convolution3D(Layer):
	
	def __init__(self, filters, kernel_size = (3,3), stride = 1, padding = 'valid', inputs = None, units = None, activation = None, initialization = None, init_params = None):
		
		Layer.__init__(self)
		
		self.id = None

		self.inputs = inputs
		self.units = units
		self.activation = activation
		self.optimizer = None #this gets assigned by model in model.build()
		
		self.fh, self.fw = kernel_size
		self.nc = filters
		self.s = stride
		self.padding = padding
		self.ph, self.pw = (None,None) 

		# setting activation methods
		self.g = getattr(activations, activation)
		self.dg = getattr(activations, 'd'+activation)

		# setting initializer
		self.initializer = Initializer(activation = activation, initialization = initialization, init_params = init_params)

		# declaring important variables            
		self.W = None
		self.b = None
		self.A = None
		self.A0 = None

		self.dW = None
		self.db = None
		
		# im2col indexes
		self.idxs = None

		# setting prediction of layer as forwardprop
		self.predict = self.forwardprop
		self.grad_check_predict = self.forwardprop

	def init(self, _id, inputs):
		
		self.id = _id
		self.inputs = inputs

		fh, fw = (self.fh, self.fw)
		s = self.s
		nc = self.nc
		padding = self.padding
		
		nh_0, nw_0, nc_0 = inputs
		
		if padding == 'valid':
			
			ph, pw = (0, 0)
			self.ph, self.pw = (0, 0)
			
		elif padding == 'same':
			
			ph = (nh_0*(s-1)+fh-s)/2
			pw = (nw_0*(s-1)+fw-s)/2
			
			assert (ph%1 == 0.0 and pw%1 == 0.0), "For 'same' padding ensure the filter size and stride are appropriate: e.g. (nw_0*(s-1)+fw-s)%2 == 0" 
			
			ph = int(ph)
			pw = int(pw)
			
			self.ph = ph
			self.pw = pw

		nh = (nh_0+2*ph-fh)/s+1
		nw = (nw_0+2*pw-fw)/s+1
		
		assert(nh%1 == 0.0 and nw%1 == 0.0), "Ensure the combination of input size, filter size and stride produce an integer output shape: e.g. ((nw_0+2*pw-fw)/s+1)%1 == 0"
		
		self.nh = int(nh)
		self.nw = int(nw)
		
		self.units = (self.nh, self.nw, self.nc)
		
		self.W = self.initializer.initialize((fh,fw,nc_0,nc))
		self.b = np.zeros((1, 1, 1, nc))
		
		self.idxs = im2col_idxs(nh_0+2*ph, nw_0+2*pw, nc_0, fh, fw, s)

	# http://cs231n.github.io/convolutional-networks/
	# matrix multiplication version

	
	def forwardprop(self, A0):

		g, W, b, fh, fw, s, ph, pw, nh, nw, nc, idxs = (self.g, self.W, self.b, self.fh, self.fw, self.s, self.ph, self.pw, self.nh, self.nw, self.nc, self.idxs)

		A0 = padding(A0, ph, pw)
		
		m, _, _, nc_0 = A0.shape
		
		A0_col = im2col(A0, fh, fw, s, idxs=idxs)

		W = W.reshape(fh*fw*nc_0,nc).T

		Z = np.zeros((m, nh, nw, nc))

		for i in range(m):

			Z[i] = np.dot(W,A0_col[i]).T.reshape(1, nh, nw, nc)

		Z += b

		A = g(Z)

		self.A = A
		self.A0_col = A0_col

		return A
	
	def backprop(self, dA0):

		A, A0_col, W, dg, idxs, nh, nw, nc, ph, pw = (self.A, self.A0_col, self.W, self.dg, self.idxs, self.nh, self.nw, self.nc, self.ph, self.pw)

		nh_0, nw_0, nc_0 = self.inputs

		m, _, _ = A0_col.shape
		
		fh, fw, nc_0, nc = W.shape

		# dZ and db
		
		dZ = dA0*self.dg(A)

		axes = tuple(i for i in range(1,len(dZ.shape)))
		db = np.sum(dZ, axis=axes, keepdims=True)

		# dW and dA_col
		W_ = W.reshape(fh*fw*nc_0, nc)
		dZ_ = np.swapaxes(dZ.reshape(m, nh*nw, nc),1,2)
	
		dW_ = np.zeros(W_.shape)
		dA_col = np.zeros(A0_col.shape)
		
		for i in range(m):

			dA_col[i] = np.dot(W_, dZ_[i])
			dW_ += np.dot(A0_col[i], dZ_[i].T)

		dW_ *= 1./m
		dW = dW_.reshape(W.shape)
	
		# flattening where features (f x f x nc_0), then training examples, are concatenated
		dA_col = np.swapaxes(dA_col,1,2).flatten()

		# dA from dA_col
		dA = np.zeros((m, nh_0+2*ph, nw_0+2*pw, nc_0)).flatten()
	
		# col2im_v2: using np.ufunc.at
		# generating indices for flattening across training examples
		idxs_m = ((np.arange(m)*(nh_0+2*ph)*(nw_0+2*pw)*nc_0)[:,None] + idxs.flatten()).flatten()
	
		# adds dA_col to the correct location in the input volume 
		# by accumulating the result for elements that are indexed multiple times
		# (fancy indexing doesn't let you accumulate)
		np.add.at(dA, idxs_m, dA_col)
	
		dA = dA.reshape(m, nh_0+2*ph, nw_0+2*pw, nc_0)

		dA = depadding(dA, ph, pw)
		
		return dA
	
	def update(self):

		W = self.W
		b = self.b
		dW = self.dW
		db = self.db

		W, b = self.optimizer.update(W, b, dW, db)

		self.W = W
		self.b = b