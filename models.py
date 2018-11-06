# models.py
# Contains the network model class

import numpy as np
from meik import utils
from meik.layers import Layer
from meik.metrics import *
from meik.normalizers import Normalizer

class Sequential:
    
    def __init__(self):
        
        self.layers = []
        self.params = {
            'learning_rate': None,
            'epochs': None,
            'loss': None,
        }
        
        self.cost = []
        self.accuracy = []
        
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
            
    def build(self, loss = None, normalization = 'none', learning_rate = 0.01, metrics = ['default'], prediction_thresholds = np.array([0.5])):
        
        th = prediction_thresholds
        
        assert(loss in ['mae', 'mse', 'binary_crossentropy', 'categorical_crossentropy']), "Provide loss as either 'mae', 'mse', 'binary_crossentropy or 'categorical_crossentropy' using kwarg 'loss'"
        
        self.params['loss'] = loss
        self.params['learning_rate'] = learning_rate
        self.params['normalization'] = normalization
        
        self.normalize = Normalizer(method = normalization)

        self.loss = getattr(utils.losses,loss)
        
        if loss == 'mae' or loss == 'mse':

            # This an annoying thing that's here to choose the default metric
            # because there isn't a single default metric for all loss functions
            metrics = (lambda metrics, loss: [loss] if metrics == ['default'] else metrics)(metrics, loss)
            
            # setting up evaluation metrics
            self.metrics = metrics_regression(metrics)

            # printing
            self.print_text = "Epoch %d/%d - "+loss+": %.4f"
            self.print_params = lambda i, epochs, cost, A, Y: (i+1, epochs, cost)
            
        elif loss == 'binary_crossentropy' or loss == 'categorical_crossentropy':
            
            metrics = (lambda metrics: ['accuracy'] if metrics == ['default'] else metrics)(metrics)
            
            # choosing evaluation metric type
            if loss == 'binary_crossentropy':
                self.metrics = metrics_binary_classification(metrics = metrics, prediction_thresholds = th)
            else:
                self.metrics = metrics_categorical_classification(metrics = metrics, prediction_thresholds = th)
            
            # printing
            self.print_text = "Epoch %d/%d - log loss: %.4f - acc: %.4f"
            acc = lambda Y, A: np.sum((A > 0.5) == Y)/Y.size
            self.print_params = lambda i, epochs, cost, A, Y: (i+1, epochs, cost, acc(Y, A))

        # TO DO: proper optimizer objects passed to layer
        for i in range(len(self.layers)):
            self.layers[i].learning_rate = learning_rate
    
    def predict(self,X):
        
        layers = self.layers
        
        A = X
        for i in range(len(layers)):
            A = layers[i].forwardprop(A)
        
        return A

    def backprop(self, Y, A):
        
        layers = self.layers
        
        dZ = A - Y
        dA = self.layers[-1].backprop_output(dZ)
        for i in range(len(layers)-2,-1,-1):
            dA = layers[i].backprop(dA)
            layers[i].update()
            
    def train(self, X, Y, epochs=1, verbose=1):
        
        layers = self.layers
        
        X_norm = self.normalize.train(X)
        
        for i in range(epochs):
            
            A = self.predict(X_norm)
            self.backprop(Y, A)
            
            cost = self.loss(Y, A)
            self.cost.append(cost)
            
            if verbose == 1:
                print(self.print_text % self.print_params(i, epochs, cost, A, Y))
            
        print("------------ Final performance ------------\n")
        print(self.print_text % self.print_params(i, epochs, cost, A, Y))     
        
    def evaluate(self, X, Y):
        
        X = self.normalize.evaluate(X)
            
        A = self.predict(X)
        score = self.metrics.evaluate(Y, A)
        
        self.score = score
        
        return score
        