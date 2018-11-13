# misc.py
# contains miscellaneous helper functions

import numpy as np
import matplotlib.pyplot as plt

def plot_training_loss(model, mode='epoch'): # or mode = 'batch'
	losses = getattr(model, mode+"_metrics")
	iters = len(losses)
	loss = np.zeros((1,iters))
	for i in range(iters):
		loss[0,i] = losses[i]['loss_tot']
	plt.plot(loss.T)