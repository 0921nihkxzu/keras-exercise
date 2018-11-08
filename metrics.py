# metrics.py
# Contains all the classes for metrics

import numpy as np
from meik.utils import losses

class Metrics:

	def __init__(self, loss = None, train_metrics = None, eval_metrics = None, thresholds = np.array([0.5])):

		assert(loss in ['mae', 'mse', 'binary_crossentropy', 'categorical_crossentropy']), "Provide loss as either 'mae', 'mse', 'binary_crossentropy or 'categorical_crossentropy' using kwarg 'loss'"

		self.loss = loss

		defaults = {
			'mae': ['mae'],
			'mse': ['mse'],
			'binary_crossentropy': ['binary_crossentropy','accuracy'],
			'categorical_crossentropy': ['accuracy']
		}

		# assigning default metric if no metric provided, else assigning provided metric
		fdef = lambda metrics, loss: defaults[loss] if metrics == None else metrics
		tm = fdef(train_metrics, loss)
		em = fdef(eval_metrics, loss)

		metrics = {
			'mae': metrics_regression,
			'mse': metrics_regression,
			'binary_crossentropy': metrics_binary_classification,
			'categorical_crossentropy': metrics_categorical_classification
		}

		self.train_metrics = metrics[loss](metrics = tm)
		self.eval_metrics = metrics[loss](metrics = em, thresholds = thresholds)

	def train(self, Y, A, reg_loss):
		
		score = self.train_metrics.evaluate(Y, A)

		if type(score) == dict:
			score['reg_loss'] = reg_loss
			score['loss_tot'] = score[self.loss] + score['reg_loss']
		elif type(score) == list:
			score[-1]['reg_loss'] = reg_loss
			score[-1]['loss_tot'] = score[-1][self.loss] + score[-1]['reg_loss']

		self.train_score = score

		return self.train_score

	def evaluate(self, Y, A, reg_loss):

		score = self.eval_metrics.evaluate(Y, A)
		if type(score) == dict:
			score['reg_loss'] = reg_loss
			score['loss_tot'] = score[self.loss] + score['reg_loss']
		elif type(score) == list:
			score[-1]['reg_loss'] = reg_loss
			score[-1]['loss_tot'] = score[-1][self.loss] + score[-1]['reg_loss']

		self.score = score
		
		return self.score

	def train_print(self, i, epochs):
		self.train_metrics.train_print(i, epochs, self.train_score, self.loss)

import numpy as np
from meik.utils.losses import mae, mse 

class metrics_regression:
	
	def __init__(self, metrics = ['mae'], thresholds = None):
		
		for metric in metrics:
			assert(metric in ['mae', 'mse', 'rmse', 'r2']), "Make sure the metrics you have provided belong to the following: 'mae', 'mse', 'rmse' or 'r2'"
		
		self.metrics = metrics

		self.mae = mae
		self.mse = mse
	
	def evaluate(self, Y, A):
				
		results = {}
		for metric in self.metrics:
			method = getattr(self, metric)
			results[metric] = method(Y,A)
		
		return results

	def train_print(self, i, epochs, results, loss):

		print_text = "Epoch %d/%d - loss_tot: %.4f - reg_loss: %.4f - "+loss+": %.4f"
		print_params = (i+1, epochs, results['loss_tot'], results['reg_loss'], results[loss])
		print(print_text % print_params)
	
	def rmse(self,y,yhat):
		return np.sqrt(mse(y,yhat))

	def r2(self,y,yhat):
		m = y.shape[1]
		SS_tot = 1./m*np.sum((y - np.mean(y))**2)
		SS_res = mse(y,yhat)
		return 1 - (SS_res/SS_tot)

import numpy as np
import matplotlib.pyplot as plt
from meik.utils.losses import binary_crossentropy

class metrics_binary_classification:
	
	def __init__(self, metrics = ['accuracy', 'binary_crossentropy'], thresholds = np.array([0.5])):
		
		for metric in metrics:
			assert(metric in ['accuracy', 'binary_crossentropy', 'precision', 'sensitivity', 'specificity', 'confusion_matrix', 'AUC', 'ROC']), "Make sure the metrics you have provided belong to the following: 'accuracy', 'binary_crossentropy', 'precision', 'sensitivity', 'specificity', 'confusion_matrix', 'AUC' or 'ROC'"
		
		self.metrics = metrics
		self.th = thresholds
		
	def evaluate(self, Y, A):

		self.populations(Y, A)

		results = {}
		for metric in self.metrics:
			method = getattr(self, metric)
			if metric == 'binary_crossentropy':
				result = method(Y, A)
			else:
				result = method()
			results[metric] = result
	
		results['accuracy'] = self.accuracy()
		results['binary_crossentropy'] = self.binary_crossentropy(Y, A)
			
		return results

	def train_print(self, i, epochs, results, loss):

		print_text = "Epoch %d/%d - loss_tot: %.4f - reg_loss: %.4f - binary_crossentropy: %.4f - acc: %.4f"
		print_params = (i+1, epochs, results['loss_tot'], results['reg_loss'], results['binary_crossentropy'], results['accuracy'])
		print(print_text % print_params)
	
	def binary_crossentropy(self, Y, A):
		# here for clarity
		return binary_crossentropy(Y, A)
		
	def populations(self, Y, A):
		
		# Calculates True Positives (TP), False Negatives (FN), True Negatives (TN) and False Positives (FP)
		# for the prediction thresholds provided by thresholds
		
		th = self.th
		n = th.shape[0]
		
		# positive and negative populations
		PP = A[Y == 1]
		NP = A[Y == 0]
		pp = len(PP)
		np_ = len(NP)

		# initializing TP, FN, TN, FP
		TP = np.zeros(n)
		TN = np.zeros(n)
		FP = np.zeros(n)
		FN = np.zeros(n)

		# Calculating the distributions for various thresholds
		for i,t in enumerate(th):
			TP[i] = np.sum(PP > t)
			TN[i] = np.sum(((NP > t)-1)*-1)
		FN = pp - TP
		FP = np_ - TN
		
		self.pp = pp
		self.np = np_
		self.TP = TP
		self.FP = FP
		self.TN = TN
		self.FN = FN
		
		return {'pp': pp, 'np': np_, 'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN}
	
	def accuracy(self):
		
		# Accuracy: probability of true positives and true negatives in total population
		
		TP = self.TP
		TN = self.TN
		pp = self.pp
		np_ = self.np
		
		return (TP + TN)/(pp + np_)

	def precision(self):
		
		# Precision: probability of a positive prediction being a true positive
		
		TP = self.TP
		FP = self.FP

		return TP/(TP + FP)
	
	def sensitivity(self):
		
		# Sensitivity: probability of true positives in true population 
		# Also known as 'Recall'
		
		TP = self.TP
		FN = self.FN

		return TP/(TP + FN)
	
	def specificity(self):
		
		# Sensitivity: probability of true negatives in true population
		
		TN = self.TN
		FP = self.FP

		return TN/(TN + FP)
		
	def confusion_matrix(self):
		
		TP = self.TP
		FP = self.FP
		TN = self.TN
		FN = self.FN
		
		# Confusion Matrix
		steps = len(TP)
		CM = np.zeros((steps,2,2))
		for i in range(steps):
			CM[i] = [[TP[i],FP[i]],
					 [FN[i],TN[i]]]
		
		return CM
		
	def AUC(self):
		
		pp = self.pp
		np_ = self.np
		TP = self.TP
		FP = self.FP
		
		# Calculating AUC
		w = (FP[:-1] - FP[1:])/np_
		h = (TP[:-1] + TP[1:])/(2*pp)
		AUC = np.dot(w,h)
		
		return AUC 
		
	def ROC(self):
		
		FP = self.FP
		TP = self.TP
		pp = self.pp
		np_ = self.np
		
		AUC_ = self.AUC()
		
		plt.figure(np.random.randint(1e9))
		plt.plot(FP/np_,TP/pp,color = 'r')
		plt.plot(np.arange(0.0,1.0,0.01), np.arange(0.0,1.0,0.01), linestyle='dashed')
		plt.xlabel('False Positives (FP) (%)')
		plt.ylabel('True Positives (TP) (%)')
		plt.legend(['Classifier - AUC: '+str(round(AUC_,2)), 'Random prediction - AUC: 0.50'])
		
		return True
	
class metrics_categorical_classification(metrics_binary_classification):
	
	def evaluate(self, Y, A):
		
		c = Y.shape[0]
		m = Y.shape[1]

		loss = losses.categorical_crossentropy(Y, A)

		results = [{} for i in range(c)]
		results.append({'categorical_crossentropy': loss})

		for i in range(c):
			
			self.populations(Y[i,:].reshape(1,m), A[i,:].reshape(1,m))

			for metric in self.metrics:
				method = getattr(self, metric)
				if metric == 'binary_crossentropy':
					result = method(Y[i,:].reshape(1,m), A[i,:].reshape(1,m))
				else:
					result = method()
				if metric == 'ROC':
					plt.title("Output "+str(i))
				results[i][metric] = result
			
		return results
		
	def train_print(self, i, epochs, results, loss):

		print_text = "Epoch %d/%d - loss_tot: %.4f - reg_loss: %.4f - categorical_crossentropy: %.4f"
		print_params = (i+1, epochs, results[-1]['loss_tot'], results[-1]['reg_loss'], results[-1]['categorical_crossentropy'])
		print(print_text % print_params)