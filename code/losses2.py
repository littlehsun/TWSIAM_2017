from keras import backend as K
from theano import tensor as T 
import numpy
from numpy.lib.stride_tricks import as_strided

def filter2(window, x):
	range1 = x.shape[0] - window.shape[0] + 1
	range2 = x.shape[1] - window.shape[1] + 1
	x1 = as_strided(x,((x.shape[0] - 10)/1 ,(x.shape[1] - 10)/1 ,11,11), (x.strides[0]*1,x.strides[1]*1,x.strides[0],x.strides[1])) * window
	res = x1.sum((2,3))
	return res

def ssim(img1, img2):
	window = numpy.array([        [0.0000,    0.0000, 0.0000, 0.0001, 0.0002, 0.0003, 0.0002, 0.0001, 0.0000, 0.0000, 0.0000],        [0.0000,    0.0001, 0.0003, 0.0008, 0.0016, 0.0020, 0.0016, 0.0008, 0.0003, 0.0001, 0.0000],        [0.0000,    0.0003, 0.0013, 0.0039, 0.0077, 0.0096, 0.0077, 0.0039, 0.0013, 0.0003, 0.0000],        [0.0001,    0.0008, 0.0039, 0.0120, 0.0233, 0.0291, 0.0233, 0.0120, 0.0039, 0.0008, 0.0001],        [0.0002,    0.0016, 0.0077, 0.0233, 0.0454, 0.0567, 0.0454, 0.0233, 0.0077, 0.0016, 0.0002],        [0.0003,    0.0020, 0.0096, 0.0291, 0.0567, 0.0708, 0.0567, 0.0291, 0.0096, 0.0020, 0.0003],        [0.0002,    0.0016, 0.0077, 0.0233, 0.0454, 0.0567, 0.0454, 0.0233, 0.0077, 0.0016, 0.0002],        [0.0001,    0.0008, 0.0039, 0.0120, 0.0233, 0.0291, 0.0233, 0.0120, 0.0039, 0.0008, 0.0001],        [0.0000,    0.0003, 0.0013, 0.0039, 0.0077, 0.0096, 0.0077, 0.0039, 0.0013, 0.0003, 0.0000],        [0.0000,    0.0001, 0.0003, 0.0008, 0.0016, 0.0020, 0.0016, 0.0008, 0.0003, 0.0001, 0.0000],        [0.0000,    0.0000, 0.0000, 0.0001, 0.0002, 0.0003, 0.0002, 0.0001, 0.0000, 0.0000, 0.0000]    ], dtype=numpy.double)

	K = [0.01, 0.03]
	L = 65535

	C1 = (K[0] * L) ** 2
	C2 = (K[1] * L) ** 2

	mu1 = filter2(window, img1)
	mu2 = filter2(window, img2)

	mu1_sq = numpy.multiply(mu1, mu1)
	mu2_sq = numpy.multiply(mu2, mu2)
	mu1_mu2 = numpy.multiply(mu1, mu2)

	sigma1_sq = filter2(window, numpy.multiply(img1, img1)) - mu1_sq
	sigma2_sq = filter2(window, numpy.multiply(img2, img2)) - mu2_sq
	sigma12 = filter2(window, numpy.multiply(img1, img2)) - mu1_mu2

	ssim_map = numpy.divide(numpy.multiply((2*mu1_mu2 + C1), (2*sigma12 + C2)), numpy.multiply((mu1_sq + mu2_sq + C1),(sigma1_sq + sigma2_sq + C2)))
	return numpy.mean(ssim_map)
def SSIM_test(y_true, y_pred,rol,col):
	H, W = img_nrows, img_ncols
	y_ture = y_ture[[:, :H-1, :W-1, :]]
	patches_true = T.nnet.neighbours.images2neibs(y_true, [4,4])
	patches_pred = T.nnet.neighbours.images2neibs(y_pred, [4,4])
	u_true = K.mean(patches_true, axis=-1)
	u_pred = K.mean(patches_pred, axis=-1)
	var_true = K.var(patches_true, axis=-1)
	var_pred = K.var(patches_pred, axis=-1)
	eps = 1e-9
	std_true = K.sqrt(var_true+eps)
	std_pred = K.sqrt(var_pred+eps)
	c1 = 0.01 ** 2
	c2 = 0.03 ** 2
	ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
	denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
	ssim /= denom #no need for clipping, c1 and c2 make the denom non-zero
	return K.mean((1.0 - ssim) / 2.0)
def calc_Dssim(img1, img2):
	#return 1/SSIM_test(img1, img2)-1
	return 1/ssim(img1, img2)-1
def feature_reconstruction_loss(base, output,row, col):
	return calc_Dssim(base, output,row,col)                                                       
#	return calc_Dssim(K.eval(K.variable(base)), K.eval(K.variable(output)))

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
