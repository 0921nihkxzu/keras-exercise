# initializations.py

# contains different weight initialization strategies

# the concept behind Glorot and He initialization has to do with an attempt to 
# maintain the variance of activations in each layer approximately constant

import numpy as np

def normal(shape, params = [0.0, 1.0], dtype=np.float64):

	assert(len(params) == 2 and all([type(l) == np.float64 or type(l) == float for l in params])), "Provide the mean and standard deviation as a list of floats [mean, std]"
	
	mean, std = params

	return np.random.standard_normal(shape).astype(dtype)*std+mean

def uniform(shape, params = [-1.0, 1.0], dtype=np.float64):
	
	assert(len(params) == 2 and all([type(l) == np.float64 or type(l) == float for l in params])), "Provide the min and max as a list of floats [min, max]"
	
	min_, max_ = params

	return np.random.uniform(min_,max_,shape).astype(dtype)

def truncated_normal(shape, params = [0.0, 1.0, 2.0], dtype=np.float64):
	
	# returns array of given shape obtained from truncated normal distribution with mean = 0, std = 1
	# method: elements are redrawn until they are within c standard deviations away from 0

	assert(len(params) == 3 and all([type(l) == np.float64 or type(l) == float for l in params])), "Provide the mean, standard deviation and cutoff as a list of floats [mean, std, c]"
	
	mean, std, c = params

	W = np.random.standard_normal(shape)
	while(np.sum(abs(W)>c)):
		new = np.random.standard_normal(shape)
		idx = (abs(W)>c)
		W[idx] = new[idx]
		
	return W.astype(dtype)*std+mean

def he_normal(shape, dtype=np.float64):

	f_in = shape[1]
	std = np.sqrt(2/f_in)
	W = truncated_normal(shape, dtype=dtype)*std

	return W

def he_uniform(shape, dtype=np.float64):

	f_in = shape[1]
	a = np.sqrt(6/f_in)
	W = np.random.uniform(-a,a,shape).astype(dtype)

	return W

def glorot_normal(shape, dtype=np.float64):

	f_out = shape[0]
	f_in = shape[1]
	std = np.sqrt(2/(f_in+f_out))
	W = truncated_normal(shape, dtype=dtype)*std

	return W

def glorot_uniform(shape, dtype=np.float64):

	f_out = shape[0]
	f_in = shape[1]
	a = np.sqrt(6/(f_in+f_out))
	W = np.random.uniform(-a,a,shape).astype(dtype)

	return W

def lecun_normal(shape, dtype=np.float64):

	f_in = shape[1]
	std = np.sqrt(1/f_in)
	W = truncated_normal(shape, dtype=dtype)*std

	return W

def lecun_uniform(shape, dtype=np.float64):

	f_in = shape[1]
	a = np.sqrt(3/f_in)
	W = np.random.uniform(-a,a,shape).astype(dtype)

	return W