# gradient_check.py

import numpy as np
from meik.models import Sequential
from meik.utils import losses

def get_params(model):
    
    theta = np.array([])
    dtheta = np.array([])
    
    for layer in model.layers:
        
        try:
            theta = np.append(theta,layer.W.flatten())
            theta = np.append(theta,layer.b.flatten())

            dtheta = np.append(dtheta,layer.dW.flatten())
            dtheta = np.append(dtheta,layer.db.flatten())

        except AttributeError:
            continue
            
    return theta.reshape(theta.shape[0],1), dtheta.reshape(dtheta.shape[0],1)

def set_params(model, theta):
    
    idx = 0
    
    for layer in model.layers:
        
        try:
            W_shape = layer.W.shape
            W_len = W_shape[0]*W_shape[1]

            W = theta[idx:idx+W_len].reshape(W_shape)

            idx += W_len

            b_shape = layer.b.shape
            b_len = b_shape[0]

            b = theta[idx:idx+b_len].reshape(b_shape)

            idx += b_len

            layer.W = W
            layer.b = b
        
        except AttributeError:
            continue
            
    return model

def J(model, theta, X, Y):
    
    # set parameters
    set_params(model, theta)
    
    # calculate loss
    loss = model.params['loss']
    Yhat = model.predict(X)
    cost = getattr(losses,loss)(Y,Yhat)
    reg_loss = model.regularization_loss()
    
    return cost+reg_loss

def num_grad(J, model, theta, X, Y, epsilon = 1e-5):

    dtheta = np.zeros(theta.shape)
    
    for i in range(len(theta)):
        
        tplus = theta.copy()   
        tplus[i] += epsilon
        
        tminus = theta.copy()
        tminus[i] -= epsilon
        
        dtheta[i] = (J(model, tplus, X, Y)-J(model, tminus, X, Y))/(2*epsilon)
    
    return np.array(dtheta).reshape(dtheta.shape[0],1)

def error(dt, dt_num):

	# calculate Euclidean distance
	dist = np.sqrt(np.sum((dt-dt_num)**2))

	# calculate normalization based on sum of vector sizes
	norm = (np.sqrt(np.sum(dt**2))+np.sqrt(np.sum(dt_num**2)))

	return dist/norm

def gradient_check(model, X, Y):

	assert(type(model) == Sequential), "Provide a Sequential model for verifying gradients"
	
	# obtain prediction
	A = model.predict(X)

	# calculate model based gradients
	model.backprop(Y, A)

	# retrieve parameters and model gradients from model
	theta, dtheta = get_params(model)

	# calculate numerical gradients
	dtheta_num = num_grad(J, model, theta, X, Y)

	# calculate error between the numerical gradients and model gradients
	res = error(dtheta, dtheta_num)

	return res

'''
# checking that set_params works
# checking set_params and get_params are inverses of each other
theta1 = theta.copy()
theta1[50] += 1
model = set_sequential_params(model, theta1)
theta2, dtheta2 = get_sequential_params(model)
np.sum(theta2 - theta)
# the results should be 1.0
'''