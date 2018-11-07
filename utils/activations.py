# activations.py
# activation functions

import numpy as np

def linear(Z):
    return Z

def sigmoid(Z):
    return 1./(1+np.exp(-Z))

def relu(Z):
    return np.maximum(np.zeros((Z.shape)),Z)

def tanh(Z):
    return np.tanh(Z)

def softmax(Z):
    Z = Z - np.amax(Z,axis=0) # to avoid overflow
    return np.exp(Z)/np.sum(np.exp(Z),axis=0)

# activation function derivatives

def dlinear(A):
    return 1

def dsigmoid(A):
    return A*(1.-A)

def drelu(A):
    return (A>0)*1.

def dtanh(A):
    return (1-A**2)

# omitted since only used at output and dL/dz provides Y-A
def dsoftmax(A,Z):
    pass
