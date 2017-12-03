import theano
import keras
from keras import backend as K
#from keras import losses as Kloss
import numpy as np
from numpy.lib.stride_tricks import as_strided
import theano.tensor as T
from keras import layers as KLayers
#import Keras_Losses as Kloss

def feature_reconstruction_loss(base, output, size):
	return K.sum(K.square(output - base))
	"""
	Feature reconstruction loss function. Encourages the 
	output img to be perceptually similar to the base image.
	"""
	size = K.variable(size)
	print("=====================ndim")
	print(K.ndim(size))
	print("======================ndim")
	#shape = (K.int_shape(base))
	base_mean = K.mean(base)
	output_mean = K.mean(output)
	base_sigma = K.sqrt(K.sum(K.square(base)-K.square(base_mean)))
	print("===========sigma")
	print(K.ndim(base_sigma))
	print("===========sigma")
	output_sigma = K.sqrt(K.sum(K.square(output)-K.square(output_mean)))
	#base_np = np.array(K.eval(base))
	#output_np = np.array(K.eval(output))
	#a, b = theano.tensor.matrices('a', 'b')
	#f = theano.function([a, b], a * b) 
	mul = ((base/255)*(output/255)) *255*255
	cov = K.sum( mul - base_mean*output_mean)
	print("==========================================cov")
	print(K.ndim(cov))
	print("======================================cov")
	#s = cov/(base_sigma*output_sigma)
	#c = 2*base_sigma*output_sigma / (K.square(base_sigma) + K.square(output_sigma))
	
	SSIM = ((2*base_mean*output_mean)*(2*cov))/ ((K.square(base_mean)+K.square(output_mean))*(K.square(base_sigma)+K.square(output_sigma)))
	DSSIM = ((1/SSIM)*size -1)
	print("============================DSSIM:")
	print(K.ndim(DSSIM*size))
	print("============================")	
	
		
	#return (1/DSSIM)**10
	


	###### ziyu
	C1 = K.variable(0)
	C2 = K.variable(100)
	C3 = K.variable(1)
	xm = base_mean
	ym = output_mean
	xs = base_sigma
	ys = output_sigma
	light = ( 2*xm*ym + C1 ) / ( K.square(xm) + K.square(ym) + C1)
	contrast = ( 2*xs*ys + C2 ) / ( K.square(xs) + K.square(ys) + C2)
	structure = ( cov + C3 ) / ( xs*ys + C3 )

	SSIM = (light**(1)) * (contrast**(1)) * (structure**(1))
	DSSIM = (1/SSIM)-1

	#return DSSIM*(1e+12)
	SE = K.sum(K.square(base-output))
	#return SSIM+SE
	#print("theano square=====")
	#print(T.sqr(base-output))
	#print("theano sqraue=====")
#	A = K.square(output-base)
#	return K.sum(A)
	#return Kloss.mean_squared_error(base, output)
	#K.sum(np.square(npoutput-base))
	MSE = SE/size
	#print(K.eval(MSE))
	PSNR = K.log( ((255*255*255)/MSE))
	#return (1/(PSNR-1))*(1e+12)
def gram_matrix(x):
	"""
	Computes the outer-product of the input tensor x.

	Input
	-----
	- x: input tensor of shape (C x H x W)

	Returns
	-------
	- x . x^T

	Note that this can be computed efficiently if x is reshaped
	as a tensor of shape (C x H*W).
	"""
	# assert K.ndim(x) == 3
	if K.image_dim_ordering() == 'th':
		features = K.batch_flatten(x)
	else:
		features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
	return K.dot(features, K.transpose(features))

def style_reconstruction_loss(base, output, img_nrows, img_ncols):
	"""
	Style reconstruction loss. Encourage the output img 
	to have same stylistic features as style image. Does not
	preserve spatial structure however.
	"""
	H, W, C = img_nrows, img_ncols, 3
	gram_base = gram_matrix(base)
	gram_output = gram_matrix(output)
	factor = 1.0 / float((2*C*H*W)**2)
	out = factor * K.sum(K.square(gram_output - gram_base))
	return out
	#return K.sum(K.square(gram_base - gram_output)) 
	base_mean = K.mean(gram_base)
	output_mean = K.mean(gram_output)
	base_sigma = K.sqrt(K.sum(K.square(gram_base)-K.square(base_mean)))
	print("===========sigma")
	print(K.ndim(base_sigma))
	print("===========sigma")
	output_sigma = K.sqrt(K.sum(K.square(gram_output)-K.square(output_mean)))
	mul = ((gram_base/255)*(gram_output/255)) *255*255
	cov = K.sum( mul - base_mean*output_mean)
	###### ziyu
	C1 = K.variable(0)
	C2 = K.variable(100)
	C3 = K.variable(1)
	xm = base_mean
	ym = output_mean
	xs = base_sigma
	ys = output_sigma
	light = ( 2*xm*ym + C1 ) / ( K.square(xm) + K.square(ym) + C1)
	contrast = ( 2*xs*ys + C2 ) / ( K.square(xs) + K.square(ys) + C2)
	structure = ( cov + C3 ) / ( xs*ys + C3 )

	SSIM = (light**(1)) * (contrast**(1)) * (structure**(1))
	DSSIM = (1/SSIM)-1

	#out = factor * K.sum(K.square(gram_output - gram_base))
#	return DSSIM*(1e+11)

def total_variation_loss(x, img_nrows, img_ncols):
	"""
	Total variational loss. Encourages spatial smoothness 
	in the output image.
	"""
	H, W = img_nrows, img_ncols
	if K.image_dim_ordering() == 'th':
		a = K.square(x[:, :, :H-1, :W-1] - x[:, :, 1:, :W-1])
		b = K.square(x[:, :, :H-1, :W-1] - x[:, :, :H-1, 1:])
	else:	
		a = K.square(x[:, :H-1, :W-1, :] - x[:, 1:, :W-1, :])
		b = K.square(x[:, :H-1, :W-1, :] - x[:, :H-1, 1:, :])

	return K.sum(K.pow(a + b, 1.25))
