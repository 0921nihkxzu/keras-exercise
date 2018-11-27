# convolution.py

# contains different utility functions for convolution layer

import numpy as np

def im2col_idxs(nh_0, nw_0, nc_0, fh, fw, s):
	
	nh = int((nh_0-fh)/s)+1
	nw = int((nw_0-fw)/s)+1
	
	filter_idxs = ((np.arange(fh)[:,None]*nw_0*nc_0 + np.arange(fw)*nc_0)[:,:,None] + np.arange(nc_0)).flatten()
	
	offset_idxs = ((np.arange(nh)*s*nw_0*nc_0)[:,None] + np.arange(nw)*s*nc_0).flatten()
	
	idxs = offset_idxs[:,None] + filter_idxs
	
	return idxs

def im2col(A0,fh,fw,s,idxs=None):

	m, nh_0, nw_0, nc_0 = A0.shape
	
	nh = int((nh_0-fh)/s)+1
	nw = int((nw_0-fw)/s)+1
	
	if type(idxs)==type(None):
		idxs = im2col_idxs(nh_0, nw_0, nc_0, fh, fw, s)
	
	# Note: arranging colums where channels are appended to each other pixel by pixel
	return np.swapaxes(A0.reshape(m,nh_0*nw_0*nc_0)[:,idxs].reshape(m,nh*nw,fh*fw*nc_0),1,2)

def padding(A, ph, pw):

	m, nh_0, nw_0, nc_0 = A.shape

	A_pad = np.zeros((m, nh_0 + 2*ph, nw_0 + 2*pw, nc_0))

	if ph != 0 and pw != 0:

		A_pad[:, ph:-ph, pw:-pw, :] = A

	elif ph == 0 and pw == 0:

		A_pad = A

	elif ph == 0 and pw != 0:

		A_pad[:, :, pw:-pw, :] = A

	elif ph != 0 and pw == 0:

		A_pad[:, ph:-ph, :, :] = A

	return A_pad


def depadding(A, ph, pw):
	
	m, nh_0, nw_0, nc_0 = A.shape
	
	A_depad = np.zeros((m, nh_0 - 2*ph, nw_0 - 2*pw, nc_0))
	
	if ph != 0 and pw != 0:

		A_depad = A[:, ph:-ph, pw:-pw, :]

	elif ph == 0 and pw == 0:

		A_depad = A

	elif ph == 0 and pw != 0:

		A_depad = A[:, :, pw:-pw, :]

	elif ph != 0 and pw == 0:

		A_depad = A[:, ph:-ph, :, :]

	return A_depad