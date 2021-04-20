
import math

#torch
import torch
import torch.nn as nn
import torch.nn.functional as F

#misc
import numpy as np

#local
from functions import graph


class DenseChebyshevGConv2d(nn.Module):
	'''
	Custom layer taking in graph nodes (torch tensor) and a normalized Laplacian (torch tensor).
	The scaled Laplacian is calculated and used to perform graph convolution with x.
	Both the scaling and the convolution happens inside the graphConvolution function.
	Weights and bias are added.

	This is meant for testing as the Laplacian needs to be sparse and not dense for efficient
	graph convolutions.
	'''

	def __init__(self, receptive_field_k, gconv_filters_in, gconv_filters_out, bias=True):
		super(DenseChebyshevGConv2d, self).__init__()

		self.gconv_filters_in = gconv_filters_in
		self.gconv_filters_out = gconv_filters_out
		self.receptive_field_k = receptive_field_k
		self.weight = nn.Parameter(torch.FloatTensor(self.gconv_filters_in*self.receptive_field_k, self.gconv_filters_out))
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
			If first pass x = (n_batch, n_features) unsqueeze s.t. x = (n_batxh, n_features, 1)
			x: (n_batch, n_features, filters) tensor
			L: (n_batch, n_batch) tensor
		'''

		n_batch, n_features, gconv_filters_in = x.shape

		x = x.permute(0, 2, 1) # (n_batch, gconv_filters_in, n_features)
		x = torch.reshape(x, (n_batch, gconv_filters_in*n_features))

		#gconv expects x = (n_batch, gconv_filters_in*n_features)
		x = graph.torchDenseChebGConv(x, L, self.receptive_field_k)
		#output = (receptive_field_k, n_batch, gconv_filters_in*n_features)


		#weight is added by x*w so change into either
		#x = (n_batch*n_features, gconv_filters_in*receptive_field_k)
		#x = (n_features*n_batch, gconv_filters_in*receptive_field_k)
		#w = (gconv_filters_in*receptive_field_k, gconv_filters_out)


		x = torch.reshape(x, (self.receptive_field_k, n_batch, gconv_filters_in, n_features))
		#x = x.permute(3, 1, 2, 0)
		x = x.permute(1, 3, 2, 0)

		#x = torch.reshape(x, (n_features*n_batch, gconv_filters_in*self.receptive_field_k))
		x = torch.reshape(x, (n_batch*n_features, gconv_filters_in*self.receptive_field_k))


		y = torch.mm(x, self.weight)#(n_batch*n_features, gconv_filters_out)

		if self.bias is not None:
			y = y + self.bias

		y = torch.reshape(y, (n_batch, n_features, self.gconv_filters_out))
		#y = y.permute(1, 0, 2)

		#DONCOMMIT add dropout to gconv layer

		return y

	def extra_repr(self) -> str:
		return 'in_features={}, out_features={}, bias={}'.format(
			self.receptive_field_k, self.gconv_filter_size, self.bias is not None
		)


class SparseChebyshevGConv2d(nn.Module):
	'''
	This custom layer takes in graph nodes and normalized Laplacian (scipy sparse matrix).
	The scaled Laplacian is calculated and used to perform graph convolution with x.
	Both the scaling and graph convolution happens inside the graphConvolution function.
	Weights and bias are added.
	'''

	def __init__(self, receptive_field_k, gconv_filters_in, gconv_filters_out, bias=True):
		super(SparseChebyshevGConv2d, self).__init__()

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

		n_batch, n_features, gconv_filters_in = x.shape

		
		###-------------------------------------------------------------------------

		#can be done in several ways:
			#deferrard: filling empty array recursively
			#or xavier: concatenating to final output recursively

		convolved_x0 = x.permute(0, 2, 1).contiguous() # (n_batch, gconv_filters_in, n_features)
		convolved_x0 = convolved_x0.view(n_batch, gconv_filters_in*n_features)
		convolved_xt = convolved_x0.unsqueeze(0) # (1, , n_batch, gconv_filters_in, n_features)

		convolved_x1 = torch.sparse.mm(L, convolved_x0) #
		convolved_xt = torch.cat((convolved_xt, convolved_x1.unsqueeze(0)), 0)
		for k in range(2, self.receptive_field_k):
			convolved_x2 = 2. * torch.sparse.mm(L, convolved_x1) - convolved_x0
			convolved_xt = torch.cat((convolved_xt, convolved_x2.unsqueeze(0)), 0)
			convolved_x0 = convolved_x1
			convolved_x1 = convolved_x2

		#out = (receptive_field_k, n_batch, g_conv_filters_in*n_features)

		###-------------------------------------------------------------------------

		#weight is added by x*w so change into either
		#x = (n_batch*n_features, gconv_filters_in*receptive_field_k)
		#x = (n_features*n_batch, gconv_filters_in*receptive_field_k)
		#w = (gconv_filters_in*receptive_field_k, gconv_filters_out)

		x = convolved_xt.view(self.receptive_field_k, n_batch, gconv_filters_in, n_features)

		#x = x.permute(3, 1, 2, 0)
		x = x.permute(1, 3, 2, 0).contiguous()

		#x = torch.reshape(x, (n_features*n_batch, gconv_filters_in*self.receptive_field_k))
		x = x.view(n_batch*n_features, gconv_filters_in*self.receptive_field_k)

		#adding weight v1
		###-------------------------------------------------------------------------

		#y = torch.mm(x, self.weight)#(n_batch*n_features, gconv_filters_out)

		#if self.bias is not None:
			#y = y + self.bias

		#y = torch.reshape(y, (n_batch, n_features, self.gconv_filters_out))
		#y = y.permute(1, 0, 2)


		#adding weight v2: remember to use transpose w for nn.Linear()
		###-------------------------------------------------------------------------

		x = F.linear(x, self.weight, self.bias) # (n_features*n_batch, gconv_filters_out)
		x = x.view(n_batch, n_features, self.gconv_filters_out)

		return x


class GILNet(nn.Module):
	def __init__(self, num_features, fully_connected_sizes, **kwargs):
		super(GILNet, self).__init__()
		'''
		Args:
			gconv_filter_sizes: list of output channels for graph convolution layers
			receptive_field_k: order k of chebyshev polynomial to approximate Laplacian decomp akin to receptive field
		'''

		#layer sizes
		self.num_features = num_features
		self.fully_connected_sizes = fully_connected_sizes
		self.gconv_filter_sizes = kwargs.pop('gconv_filter_sizes')
		self.receptive_field_k = kwargs.pop('receptive_field_k')
		self.pooling_sizes = kwargs.pop('pooling_sizes')

		#scalar hyperparameters
		self.regularization = kwargs.pop('regularization')
		self.dropout = kwargs.pop('dropout')
		self.learning_rate = kwargs.pop('learning_rate')
		self.decay_rate = kwargs.pop('decay_rate')
		self.momentum = kwargs.pop('momentum')

		#layers
		self.gconv1 = SparseChebyshevGConv2d(self.receptive_field_k, self.gconv_filter_sizes[0], self.gconv_filter_sizes[1])
		self.gconv2 = SparseChebyshevGConv2d(self.receptive_field_k, self.gconv_filter_sizes[1], self.gconv_filter_sizes[2])

		#self.gconv1 = DenseChebyshevGConv2d(self.receptive_field_k, self.gconv_filter_sizes[0], self.gconv_filter_sizes[1])
		#self.gconv2 = DenseChebyshevGConv2d(self.receptive_field_k, self.gconv_filter_sizes[1], self.gconv_filter_sizes[2])

		self.pool1 = nn.AvgPool1d(kernel_size=self.pooling_sizes[0], stride=self.pooling_sizes[0])
		self.pool2 = nn.AvgPool1d(kernel_size=self.pooling_sizes[1], stride=self.pooling_sizes[1])

		self.fc1 = nn.Linear(num_features*self.gconv_filter_sizes[1], fully_connected_sizes[0])
		self.fc2 = nn.Linear(fully_connected_sizes[0], fully_connected_sizes[1])

		self.activation = nn.ReLU()

	def forward(self, x, L):
		'''
		Args:
			x(torch tensor): (n_nodes, n_features)
			L(sparse tensor): UNSCALED  graph Laplacian (n_nodes, n_nodes)
		'''

		#n_batch, n_features = x.shape

		#x = self.activation(self.graphconv1(x))
		#x = self.graphpool1(x)

		#x = self.activation(self.graphconv2(x))
		#x = self.graphpool2(x)

		#class to node relationship regression module

		#softmax


		#add gconv_filters_in=1 before first gconv 
		x = torch.unsqueeze(x, 2) #(n_nodes, n_features, 1)

		x = self.gconv1(x, L[0]) #(n_nodes, n_features, self.gconv_filters_out[0])
		x = self.activation(x)

		x = x.permute(1, 2, 0).contiguous() # (n_features, gconv_filters_out, n_nodes)
		x = self.pool1(x) # (n_features, gconv_filters_out, n_nodes/p)
		x = x.permute(2, 0, 1).contiguous() # (n_nodes, n_features, gconv_filters_out[0]/pooling_size[0])

		x = self.gconv2(x, L[2]) #(n_nodes, n_features, self.gconv_filters_out[1])
		x = self.activation(x)

		x = x.permute(1, 2, 0).contiguous() # (n_features, gconv_filters_out[1], n_nodes)
		x = self.pool2(x) # should be (n_features, g_conv_filters_out[1], n_nodes/pooling_size[1])
		x = x.permute(2, 0, 1).contiguous() # (n_nodes/pooling_size[1], n_features, gconv_filters_out[1])

		x = x.view(-1, x.shape[1]*x.shape[2]) # (n_nodes/pooling_size[1], n_features*gconv_filters_out[1])

		print(x.shape)
		exit()

		x = self.fc1(x)
		print(x.shape)
		exit()
		x = self.fc2(x)
		
		return x







