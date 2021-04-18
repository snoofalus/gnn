
import math

#torch
import torch
import torch.nn as nn
import torch.nn.functional as F

#misc
import numpy as np

#local
from functions import graph


class GConv2d(nn.Module):
	'''
	This custom layer takes in graph nodes (torch tensor) and normalized Laplacian (torch tensor).
	The scaled Laplacian is calculated and used to perform graph convolution with x.
	Both the scaling and graph convolution happens inside the graphConvolution function.
	Weights and bias are added.
	'''

	def __init__(self, receptive_field_k, gconv_filters_in, gconv_filters_out, bias=True):
		super(GConv2d, self).__init__()

		self.gconv_filters_in = gconv_filters_in
		self.gconv_filters_out = gconv_filters_out
		self.receptive_field_k = receptive_field_k
		self.weight = nn.Parameter(torch.FloatTensor(self.gconv_filters_out, self.gconv_filters_in*self.receptive_field_k))
		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(self.gconv_filters_out))
		else: 
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		'''
		General convention to start weights in range [-y, y] 
		where y = 1 / sqrt(n), n is inputs to a neuron
		but trying kaiming because of nonlinear activation afterwards
		'''
		nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, x, L):
		'''
		Args: 
			x: (n_batch, n_features, filters) torch tensor
			L: (n_batch, n_batch) torch tensor
		'''

		n_batch, n_features, gconv_filters_in = x.shape

		x = x.permute(0, 2, 1) # (n_batch, gconv_filters_in, n_features)
		x = torch.reshape(x, (n_batch, gconv_filters_in*n_features))

		#gconv expects x = (n_batch, gconv_filters_in*n_features)
		x = graph.torchGraphConvolution(x, L, self.receptive_field_k)
		#output = (receptive_field_k, n_batch, gconv_filters_in*n_features)


		#weight is added by x*w.T so change into either
		#x = (n_batch*n_features, gconv_filters_in*receptive_field_k)
		#x = (n_features*n_batch, gconv_filters_in*receptive_field_k)
		#w = (gconv_filters_out, gconv_filters_in*receptive_field_k)


		x = torch.reshape(x, (self.receptive_field_k, n_batch, gconv_filters_in, n_features))
		#x = x.permute(3, 1, 2, 0)
		x = x.permute(1, 3, 2, 0)

		#x = torch.reshape(x, (n_features*n_batch, gconv_filters_in*self.receptive_field_k))
		x = torch.reshape(x, (n_batch*n_features, gconv_filters_in*self.receptive_field_k))

		y = F.linear(x, self.weight, self.bias) # (n_features*n_batch, gconv_filters_out)
		y = torch.reshape(y, (n_batch, n_features, self.gconv_filters_out))
		#y = y.permute(1, 0, 2)

		#DONCOMMIT add dropout to gconv layer


		return y

	def extra_repr(self) -> str:
		return 'in_features={}, out_features={}, bias={}'.format(
			self.receptive_field_k, self.gconv_filter_size, self.bias is not None
		)


class SparseGConv2d(nn.Module):
	'''
	This custom layer takes in graph nodes and normalized Laplacian (scipy sparse matrix).
	The scaled Laplacian is calculated and used to perform graph convolution with x.
	Both the scaling and graph convolution happens inside the graphConvolution function.
	Weights and bias are added.
	'''

	def __init__(self, receptive_field_k, gconv_filter_size, bias=True):
		super(SparseGConv2d, self).__init__()

		self.gconv_filter_size = gconv_filter_size
		self.receptive_field_k = receptive_field_k
		self.weight = nn.Parameter(torch.FloatTensor(gconv_filter_size, receptive_field_k))
		if bias:
			self.bias = nn.Parameter(torch.FloatTensor(gconv_filter_size))
		else: 
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		'''
		General convention to start weights in range [-y, y] 
		where y = 1 / sqrt(n), n is inputs to a neuron
		but trying kaiming because of nonlinear activation afterwards
		'''
		nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
		if self.bias is not None:
			fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
			bound = 1 / math.sqrt(fan_in)
			nn.init.uniform_(self.bias, -bound, bound)

	def forward(self, x, L):

		n_batch, n_features = x.shape

		#(receptive_field_k, n_batch, n_features)
		x = graph.sparseGraphConvolution(x, L, self.receptive_field_k)
		print(x.shape)
		exit()
		x = torch.from_numpy(x).float() #DONCOMMIT try to implement gconv fully in torch 

		#weight is added by x*w.T = (n_batch*n_features, k)*(gconv_filter_size, k).T
		#so need to transpose and reshape 

		#not sure if it should be (n_batch, n_features) or (n_features, n_batch)
		x = x.permute(1, 2, 0)
		x = torch.reshape(x, (n_batch*n_features, self.receptive_field_k))

		#(n_batch*n_features, receptive_field_k)
		y = F.linear(x, self.weight, self.bias)
		y = torch.reshape(y, (n_batch, n_features, self.gconv_filter_size))


		return y


class GILNet(nn.Module):
	def __init__(self, fully_connected_sizes, sparse_W, **kwargs):
		super(GILNet, self).__init__()
		'''

		Args:
			gconv_filter_sizes: list of output channels for graph convolution layers
			receptive_field_k: order k of chebyshev polynomial to approximate Laplacian decomp akin to receptive field
		'''

		#layer sizes
		#self.sparse_W = kwargs.pop('sparse_W')
		self.sparse_W = sparse_W
		self.gconv_filter_sizes = kwargs.pop('gconv_filter_sizes')
		self.receptive_field_k = kwargs.pop('receptive_field_k')
		self.pooling_sizes = kwargs.pop('pooling_sizes')
		self.fully_connected_sizes = fully_connected_sizes

		#scalar hyperparameters
		self.regularization = kwargs.pop('regularization')
		self.dropout = kwargs.pop('dropout')
		self.learning_rate = kwargs.pop('learning_rate')
		self.decay_rate = kwargs.pop('decay_rate')
		self.momentum = kwargs.pop('momentum')

		#layers

		if self.sparse_W:

			self.gconv1 = SparseGConv2d(self.receptive_field_k, self.gconv_filter_sizes[0], self.gconv_filter_sizes[1])
			self.gconv2 = SparseGConv2d(self.receptive_field_k, self.gconv_filter_sizes[0], self.gconv_filter_sizes[1])

		else:

			self.gconv1 = GConv2d(self.receptive_field_k, self.gconv_filter_sizes[0], self.gconv_filter_sizes[1])
			self.gconv2 = GConv2d(self.receptive_field_k, self.gconv_filter_sizes[1], self.gconv_filter_sizes[2])

		self.pool1 = nn.AvgPool1d(kernel_size=self.pooling_sizes[0], stride=self.pooling_sizes[0], padding=1)#1
		self.pool2 = nn.AvgPool1d(kernel_size=self.pooling_sizes[1], stride=self.pooling_sizes[1], padding=0)#0

		self.fc1 = nn.Linear()

		#self.pool1 = nn.AvgPool1d(kernel_size=[self.pooling_sizes[1],1,1,1], stride=[self.pooling_sizes[1],1,1,1], padding='SAME')

		self.activation = nn.ReLU()

	def forward(self, x, L):
		'''
		Args:
			x: numpy array batch of graph nodes shape (n_nodes, n_features)
			L: UNSCALED graph laplacian (n_nodes, n_nodes)
		'''

		#n_batch, n_features = x.shape

		#x = self.activation(self.graphconv1(x))
		#x = self.graphpool1(x)

		#x = self.activation(self.graphconv2(x))
		#x = self.graphpool2(x)

		#class to node relationship regression module

		#softmax


		#add gconv_filters_in=1 before first gconv 
		x = torch.unsqueeze(x, 2)


		x = self.gconv1(x, L[0])
		x = self.activation(x)

		x = x.permute(1, 2, 0) # (n_features, n_nodes, gconv_filters_out)
		print(x.shape)
		x = self.pool1(x) # should be (n_features, n_nodes, gconv_filters_out/pooling_size[0])
		print(x.shape)
		x = x.permute(2, 0, 1) # (n_nodes, n_features, gconv_filters_out/pooling_size[0])


		x = self.gconv2(x, L[2])
		x = self.activation(x)

		x = x.permute(1, 2, 0) # (n_features, n_nodes, gconv_filters_out)
		print(x.shape)
		x = self.pool2(x) # should be (n_features, n_nodes, gconv_filters_out/pooling_size[1])
		print(x.shape)
		x = x.permute(2, 0, 1) # (n_nodes, n_features, gconv_filters_out/pooling_size[1])

		return x







