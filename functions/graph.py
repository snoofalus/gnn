#misc
import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.sparse

import torch

def weightedGraph(data_modality):
	#reshape into x datapoints with y dimensions
	C, H, W = data_modality.shape

	#total nodes
	n = H*W

	#pdist takes data in (n points, m dim space)
	data_flatten = np.reshape(data_modality, (C, n))
	data = np.transpose(data_flatten)

	#distance vector elements = (n-1) + (n-2)+...
	v = pdist(data, 'euclidean')

	#similarity matrix W
	W = squareform(v)

	#scale sets to make distances comparable
	W = scale(W)

	return W

def scale(W):

	scaling = np.std(W)

	W = W/scaling

	return W

def fuseMatrix(W1, W2):

	W = np.maximum(W1, W2)

	#W = np.exp(-W)

	return W

def graphKNN(d, k):

	# k-NN graph.
	idx = np.argsort(d)[:, 1:k+1]
	d.sort()
	d = d[:, 1:k+1] #this is a vector 100x10, nxk
	return d, idx

def adjacency(dist, idx):
	"""Return the adjacency matrix of a kNN graph."""
	M, k = dist.shape
	assert M, k == idx.shape
	assert dist.min() >= 0

	# Weights.
	#sigma2 = np.mean(dist[:, -1])**2
	#dist = np.exp(- dist**2 / sigma2)

	dist = np.exp(-dist)

	# Weight matrix.
	I = np.arange(0, M).repeat(k)
	J = idx.reshape(M*k)
	V = dist.reshape(M*k)
	W = scipy.sparse.coo_matrix((V, (I, J)), shape=(M, M))

	# No self-connections.
	W.setdiag(0)

	#W = scipy.sparse.coo_matrix.todense(W)
	#print(W[0, :100])
	#print(W[:100,0])

	#exit()

	# Non-directed graph.
	bigger = W.T > W
	W = W - W.multiply(bigger) + W.T.multiply(bigger)

	assert W.nnz % 2 == 0
	assert np.abs(W - W.T).mean() < 1e-10
	assert type(W) is scipy.sparse.csr.csr_matrix
	return W

def numpyNormalizedLaplacian(W):
	'''
	Normalized graph laplacian.
	The symmetric graph Laplacian is defined as

		L_sym = I - D^(-1/2)*W*D^(-1/2)

	and is used as input to the chebyshev graph convolution.

	Args:
		W(scipy sparse): numpy weighted graph similarity matrix 

	Outputs:
		L(scipy sparse): normalized graph Laplacian 
	'''

	#d_i = sum_j w_ij
	d = np.sum(W, axis = 0)

	#Included in several similar algos for numerical stability in normalized sym laplacian 
	#but doesnt seem to have effect. Possibly more important for sparse W.
	d += np.spacing(np.array(0, W.dtype))

	D_neg = np.diag(1 / (np.sqrt(d)))
	I = np.eye(W.shape[0])

	L_sym = I - np.linalg.multi_dot([D_neg, W, D_neg])

	L_sym = scipy.sparse.csr_matrix.todense(L_sym)

	return L_sym

def scipyNormalizedLaplacian(W, out):
	'''
	Normalized graph laplacian.
	The symmetric graph Laplacian is defined as

		L_sym = I - D^(-1/2)*W*D^(-1/2)

	and is used as input to the chebyshev graph convolution.

	Args:
		W(scipy sparse): weighted graph similarity matrix

	Outputs:
		L: (scipy sparse) normalized graph Laplacian
	'''

	#d_i = sum_j w_ij
	d = scipy.sparse.csr_matrix.sum(W, axis = 0)#returns np matrix

	#Included in several similar algos for numerical stability in normalized sym laplacian 
	#but doesnt seem to have effect. Possibly more important for sparse W.
	d += np.spacing(np.array(0, W.dtype))

	d = np.squeeze(np.asarray(d))

	D_neg = scipy.sparse.diags(1 / (np.sqrt(d)))

	I = scipy.sparse.eye(W.shape[0], dtype=W.dtype)

	L_sym = I - D_neg*W*D_neg

	
	if out == 'dense':
		L_sym = scipy.sparse.csr_matrix.todense(L_sym)
		L_sym = torch.from_numpy(L_sym).float()

	elif out == 'sparse':
		L_sym = scipy.sparse.csr_matrix.tocoo(L_sym)

	#L_sym = scipy.sparse.csr_matrix.tocoo(L_sym)	

	return L_sym

def torchNormalizedLaplacian(W):
	d = scipy.sparse.csr_matrix.sum(W, axis = 0)#returns np matrix

	#Included in several similar algos for numerical stability in normalized sym laplacian 
	#but doesnt seem to have effect. Possibly more important for sparse W.
	d += np.spacing(np.array(0, W.dtype))

	d = np.squeeze(np.asarray(d))

	D_neg = scipy.sparse.diags(1 / (np.sqrt(d)))

	I = scipy.sparse.eye(W.shape[0], dtype=W.dtype)

	L_sym = I - D_neg*W*D_neg

	L_sym = scipy.sparse.csr_matrix.tocoo(L_sym)


	#scipy to torch
	row = torch.from_numpy(L_sym.row.astype(np.int64)).to(torch.long)
	col = torch.from_numpy(L_sym.col.astype(np.int64)).to(torch.long)
	edge_index = torch.stack([row, col], dim=0)

	val = torch.from_numpy(L_sym.data.astype(np.float64)).to(torch.float)

	out = torch.sparse.FloatTensor(edge_index, val, torch.Size(L_sym.shape))

	return out

def numpyGraphConvolution(X, L, receptive_field_k):
	'''
	Performs graph convolution on features X of nodes around subgraph G_vi s.t.
	conv(X) = g_theta(L) conv with X = U*g_theta(Lambda_hat)*U^T*X
	where g_theta is an approximation using chebyshev polynomial of order k.
	This will encode the local representation around v_i and can be used in combination 
	with representation around node v_j to calculate similarity score between nodes
	v_i and v_j.

	Chebyshev polynomial computed by recurrence relation
	T_k(x) = 2*x*T_{k-1}(x) - T_{k-2}(x)
	T_0 = 1
	T_1 = x

	s.t. 
	x_bar0 = 1*X
	x_bar1 = L_tilde*X
	x_bark = 2*x*T_(k-1)(x) - T_(k-2)(x)

	Args:
		X: Number of nodes by feature dimensions (n, d)  of the local subgraph 
		L: Normalized graph Laplacian (n, n) L = I - D^(-1/2)*E*D^(-1/2) of the local subgraph
		receptive_field_k: order k of chebyshev polynomial around the scaled eigenvalue matrix Lambda_hat

	Output:
		convolved_xk: signal X convolved with chebyshev polynomial of k'th order around L_tilde
	'''

	n_nodes, feature_dimensions = X.shape
	
	#2 for normalized laplacian
	lambda_max = 2
	L_tilde = numpyScaledLaplacian(L, lambda_max)

	convolved_xk = np.zeros((receptive_field_k, n_nodes, feature_dimensions))

	convolved_xk[0, :, :] = X
	convolved_xk[1, :, :] = np.matmul(L, X)

	for k in range(2, receptive_field_k):
		convolved_xk[k, :, :] = 2*np.matmul(L, convolved_xk[k-1, :, :]) - convolved_xk[k-2, :, :]

	return convolved_xk

def torchDenseChebGConv(X, L, receptive_field_k):
	'''
	Performs graph convolution on features X of nodes around subgraph G_vi s.t.
	conv(X) = g_theta(L) conv with X = U*g_theta(Lambda_hat)*U^T*X
	where g_theta is an approximation using chebyshev polynomial of order k.
	This will encode the local representation around v_i and can be used in combination 
	with representation around node v_j to calculate similarity score between nodes
	v_i and v_j.

	Chebyshev polynomial computed by recurrence relation
	T_k(x) = 2*x*T_{k-1}(x) - T_{k-2}(x)
	T_0 = 1
	T_1 = x

	s.t. 
	x_bar0 = 1*X
	x_bar1 = L_tilde*X
	x_bark = 2*x*T_(k-1)(x) - T_(k-2)(x)

	Args:
		X(tensor): Number of nodes by feature dimensions (n, d)  of the local subgraph 
		L(tensor): Normalized graph Laplacian (n, n) L = I - D^(-1/2)*E*D^(-1/2) of the local subgraph
		receptive_field_k(int): order k of chebyshev polynomial around the scaled eigenvalue matrix Lambda_hat

	Output:
		convolved_xk(tensor): signal X convolved with chebyshev polynomial of k'th order around L_tilde
	'''

	n_nodes, feature_dimensions = X.shape

	#2 for normalized laplacian
	lambda_max = 2.
	L_tilde = torchScaledLaplacian(L, lambda_max)

	convolved_xk = torch.empty((receptive_field_k, n_nodes, feature_dimensions))

	convolved_xk[0, :, :] = X
	convolved_xk[1, :, :] = torch.mm(L, X)

	for k in range(2, receptive_field_k):
		convolved_xk[k, :, :] = torch.mul(torch.mm(L, convolved_xk[k-1, :, :]), 2.) - convolved_xk[k-2, :, :]

	return convolved_xk

def numpyScaledLaplacian(L, lambda_max):
	'''
	The Chebyshev point of evaluation used as part of input to 
	the graph convolution, i.e.
		L_tilde = 2*L/eigval_max - I
	Depending on lambda_max this function may not be needed.
	'''	

	I = np.eye(L.shape[0])

	#lambda_max will be 2 for normalized laplacian, making this unnecessary
	L_tilde = (2*L) / lambda_max - I

	return L_tilde

def torchScaledLaplacian(L, lambda_max):
	'''
	The Chebyshev point of evaluation used as part of input to 
	the graph convolution, i.e.
		L_tilde = 2*L/eigval_max - I
	Depending on lambda_max this function may not be needed.
	'''	

	#lambda_max will be 2 for normalized laplacian, making this unnecessary
	L_tilde = torch.mul(L, torch.div(2, lambda_max)) - torch.eye(L.shape[0])

	return L_tilde

def calculatePathReachability(reference_node, query_node, W, max_steps):
	'''
	Finds a vector of probabilites (max_steps,) where element t is the probabiliy 
	of going from node i to node j in t steps.

	the probability matrix where P_ij is the probability of going from 
	vertex i to vertex j in one step is defined as 
		P = D^(-1)*W
		D_ii = sum_ii W_ij 
	where D_ii are the elements of the diagonal degree matrix.

	The probability of reaching vertex j from vertex i in t-steps is
		P_ij for t=1
		sum_h P_ih P_(hj)^(t-1)

	Args:
		reference_node: idx of labeled reference node v_i

		query_node: idx of unlabeled query node v_j 

		W: Full adjacency matrix between reference and query nodes(as 
		compared to the subgraphs in node representation).

		max_steps: the largest possible transition step t to calculate

	Output:
		reachability_vector: vector of probabilities for reaching query_node
		from reference_node in t steps.
	'''

	n_nodes = W.shape[0]

	#d_i = sum_j w_ij
	d = np.sum(W, axis = 0)

	D_inv = np.diag(1 / d)

	#transition probability matrix with elements P_ij
	P = np.matmul(D_inv, W)

	reachability_vector = np.zeros((max_steps, ))

	#P_ij^t = P_ij if t=1
	reachability_vector[0] = P[reference_node, query_node]

	for t in range(2, max_steps + 1):

		#probability of going from i to j in t steps
		P_ij_step = 0

		for h in range(n_nodes):
			#P_ij^t = sum_h P_ih*Phj^(t-1)
			P_ij_step += P[reference_node, h] * P[h, query_node]**(t-1)

		reachability_vector[t-1] = P_ij_step

	return reachability_vector

def calculateClassToNodeRelationship(reachability_vector):
	'''
	Gives a class relationship between nodes vi, vj, by
	considering both path reachability between nodes and 
	local representations around the nodes found by graph 
	convolution on their respective subgraphs.

		f_P(vi, vj, W): nodePathReachability 

		phi_w(f_P): function mapping reachability into a weight 
		value w_ij, done by using two 16 dim fully connected
		layers.

	The weight value is further used to weight the local features 
	at node vi, f_e(G_vi). Finally, similarity score between two nodes
	are found by concatenating weighted features of reference node vi 
	with unweighted features of query node and adding a fully connected
	layer with C dimensions. Also called relation regression score.


	'''
	x = 10 #DONCOMMIT currently this func is just a placeholder
	return x



