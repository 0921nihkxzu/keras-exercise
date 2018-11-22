# misc.py
# contains miscellaneous helper functions

import numpy as np
import matplotlib.pyplot as plt

def plot_training_loss(model, loss='binary_crossentropy', mode='epoch'): # or mode = 'batch'
	
	losses = getattr(model, mode+"_metrics")
	iters = len(losses)
	
	if loss == 'binary_crossentropy':
		getloss = lambda i: losses[i]['loss_tot']
	elif loss == 'categorical_crossentropy':
		getloss = lambda i: losses[i][-1]['loss_tot']

	loss = np.zeros((1,iters))
	for i in range(iters):
		loss[0,i] = getloss(i)

	plt.plot(loss.T)