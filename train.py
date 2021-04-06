
#builtin
import argparse
import os
from itertools import combinations

#torch
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim

#misc
import numpy as np
import matplotlib.pyplot as plt

#local
#import dataset.s1s2glcm as dataset
#import dataset.houston as dataset
import dataset.trento as dataset

from functions import graph, coarsening, gilnet


parser = argparse.ArgumentParser(description='PyTorch Graph Inference Learning')

parser.add_argument('--batch-size', default=32, type=int, metavar='N',
					help='train batchsize')

parser.add_argument('--n-classes', default=16, type=int, metavar='N',
					help='number of classes')


#graph conv and pooling hyperparameters

parser.add_argument('--gconv-filter-sizes', default=[1, 128, 256], type=list,
					help='list of output channels for graph convolution layers')

parser.add_argument('--receptive-field-k', default=2, type=int, metavar='N',
					help='order k of chebyshev polynomial to approximate Laplacian decomp akin to receptive field')

parser.add_argument('--pooling_sizes', default=[4, 2], type=list,
					help='after graph coarsening regular 1D mean or max pooling can be performed on graph signals')

	#gconv_filter_sizes = [128, 256] #n graphconv fitlers
	#receptive_field_k = 2
	#pooling_sizes = [4, 2]
	#fully_connected_sizes = [16, n_classes]

#fully connected hyperparameters
parser.add_argument('--regularization', default=5e-4, type=float,
					help='adds a penalty term to the error function to reduce overfitting')

parser.add_argument('--dropout', default=1, type=float,
					help='chance of ignoring neurons in fwrd bkwrd pass during training to reduce overfitting')

parser.add_argument('--learning-rate', default=1e-3, type=float,
					help='hyperparameter controlling how much weights are changed in backpropagation')

parser.add_argument('--decay-rate', default=0.95, type=float,
					help='decrease learning rate over iterations')

parser.add_argument('--momentum', default=0.9, type=float,
					help='smooth progression of accuracy to accelerate learning')

args = parser.parse_args()
###
#ALGORITHM 1 GRAPH DEFINITIOn:
###-------------------------------------------------------------------------
'''
NEED GRAPH REPRESENTATION of Image, G = (V, E, X) which can be done by
-As LP with Fully Convolutional Net training supervised to learn node features X
-As Graph fusion with exp of distance max merge

'''
###
#ALGORITHM 2 NODE REPRESENTATION:
###-------------------------------------------------------------------------
'''
Local representation at vertex v_i extracted by performing graph conv on
subgraph G_{v_i}




normalized graph Laplacian matrix 
L = I_n - D^{-1/2}*E*D^{-1/2} =>eigdecom=> U*Lambda*U^T

Use convolution on X, conv(X) = U*g_theta(Lambda)*U^T*X, theta is fourier coefficients

g_theta(Lambda) =  sum_{k=0}^{K-1} theta_k*T_k(Lambda_hat)
T_j(Lambda_hat) Chebyshev polynomial of order k evaluated at scaled Laplacian
L_hat = 2*L/lambda_max - I_n
Lambda_hat = 2*Lambda/lambda_max - I_n
'''

###
#ALGORITHM 3 PATH REACHABILITY:
###-------------------------------------------------------------------------
'''
Probabilities of paths from vertex i to vertex j by random walks on graphs
by probability matrix P

P = D^{-1}*E
D is diagonal degree  matrix of E.

P_{ij}^t = P_ij if t = 1, sum_h P_{ih}P_{h,j}^{t-1} if t>1


Write node reachability from v_i to v_j as d_p dim vector
f_P(v_i, v_j, E) = [P_{ij}, P_{ij}^2,...,P_{ij}^d_p]
d_p refers to step length of longest path from v_i to v_j
'''

###
#ALGORITHM 4 PATH REACHABILITY CLASS TO NODE REACHABILITY:
###-------------------------------------------------------------------------
'''
1. Paper uses two fully connected 16 to map path reachability f_P into a weight value.
2. Use weight value w_{i->j} to weight local features at node v_i, f_e(G_{v_i})
3. Concatenate weighted features of v_i and local features of v_j. Use one fully connected layer with C dimensions to get relation regression score
'''


###
#Houston
###-------------------------------------------------------------------------

'''
data_directory = os.path.join(root, 'images/houston/by-image/data')
mask_directory = os.path.join(root, 'images/houston/by-image/mask')


houston_dataset = dataset.get_houston(root)
loader = data.DataLoader(houston_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

#image1, mask1 = houston_dataset[390]#360


#create subplot with 6-9 plots to find good images


#groundtruth = np.load(os.path.join(root, 'workdir/houston/houston_groundtruth.npy')).T
#fig=plt.figure()
#plt.imshow(groundtruth)
#plt.show()
#exit()

norm = np.load(os.path.join(root,'workdir/houston/houston_norm.npy'))


image1 = np.load(os.path.abspath('data1.npy'))

channels, _, _ = image1.shape

for ch in range(channels):
	tmp = image1[ch, :, :] 
	image1[ch, :, :] = (tmp - norm[ch, 0])/norm[ch,1]


#add small constant for numerical stability
image1 = image1 + 0.1

mask1 = np.load(os.path.abspath('mask1.npy'))

arr1 = image1[:7, :, :]
arr2 = image1[7:, :, :]
'''

###
#Trento
###-------------------------------------------------------------------------

root = os.path.abspath('../data-local')

data_directory = os.path.join(root, 'images/trento/by-image/data')
mask_directory = os.path.join(root, 'images/trento/by-image/mask')


trento_dataset = dataset.get_trento(root) #len 48 if 64x64
loader = data.DataLoader(trento_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)


image1, mask1 = trento_dataset[1]#360


#create subplot with 6-9 plots to find good images

#fig=plt.figure()
#plt.imshow(mask1)
#plt.show()
#exit()



#add small constant for numerical stability
image1 = image1 + 0.1


###
#Random 
###-------------------------------------------------------------------------

#arr1 = np.random.rand(2, 16, 16)
#arr2 = np.random.rand(13, 16, 16)
#mask1 = np.random.randint(0, 6, (16, 16))

#arr1 = np.random.rand(7, 32, 32)
#arr2 = np.random.rand(144, 32, 32)
#mask1 = np.random.randint(0, 3, (32, 32))

#arr1 = np.random.rand(7, 64, 64)
#arr2 = np.random.rand(144, 64, 64)
#mask1 = np.random.randint(0, 5, (64, 64))

#120 worked
#arr1 = np.random.rand(7, 120, 120)
#arr2 = np.random.rand(144, 120, 120)
#mask1 = np.random.randint(0, 6, (120, 120))

###-------------------------------------------------------------------------

def main(image, mask, args):

	mode1 = image[:7, :, :]
	mode2 = image[7:, :, :]

	#each mode builds one graph of node distances 
	W1 = graph.weightedGraph(mode1)
	W2 = graph.weightedGraph(mode2)

	#fuse graphs by max and take exp
	W = graph.fuseMatrix(W1, W2)

	W = W[:100, :100]

	#use this and comment out below to use np L
	#sparse_W = False
	#W_sub = W[0:100, 0:100]
	

	#use this and comment out above to use sparse L
	#dist, idx = graph.graphKNN(W, 1000)
	dist, idx = graph.graphKNN(W, 10)
	A = graph.adjacency(dist, idx)
	sparse_W = False

	#largest should be 4104, 8 added



	graphs, perm = coarsening.coarsen(A, levels=3, self_connections=False)


	#collapse into (n_features, height*width)
	X_train = np.reshape(image, (image.shape[0], -1))

	X_train = X_train[:, :100]

	X_train = coarsening.perm_data(X_train, perm)

	X_train = torch.from_numpy(X_train.T).float()

	if sparse_W:
		L = [graph.normalizedLaplacian(A, out='sparse') for A in graphs]

	else:
		L = [graph.normalizedLaplacian(A, out='dense') for A in graphs]



	#if type(W) is np.ndarray:
	#elif type(W) is scipy.sparse.csr.csr_matrix:


	#t = 3
	#reachability_vector = graph.calculatePathReachability(3, 20, W_sub, t)

	#relation_regression_score = graph.calculateClassToNodeRelationship(reachability_vector)

	#print(reachability_vector[2])


	n_classes = len(np.unique(mask))
	fully_connected_sizes = [16, n_classes]

	params = dict(vars(args))
	model = gilnet.GILNet(fully_connected_sizes, sparse_W, **params)

	criterion = nn.CrossEntropyLoss()

	#optimizer = optim.Adam(model.parameters(), lr=args.lr)
	optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
	optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.decay_rate)

	output = model(X_train, L)

	



main(image1, mask1, args)