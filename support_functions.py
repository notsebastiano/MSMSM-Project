import numpy as np
import matplotlib.pyplot as plt
import prody

from prody.dynamics.compare import calcSubspaceOverlap

import MDAnalysis as mda

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities


from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.neighbors import NearestNeighbors
from scipy import spatial
#from sklearn.metrics.pairwise import cosine_similarity
import math
from scipy.spatial.distance import minkowski


#############

def clusters_from_labels(labels,calphas,mapping_data = 0,mapping = False):
	'''
	- "labels" has form:[l1,l1,l2,l3,l2,l3,l3,...]
		such as entry i tells that atom i is in cluster l_j
	- returns:
		- clusters_pos[i][0] is the average position on x axis for cluster i
		- cluster_masses: entry i is mass of cluster in clusters_pos[i]
	'''
	found = []
	clusters = {}
	for i,l in enumerate(labels):
		if l in found:
			tmp = clusters[str(l)]
			clusters[str(l)] = np.concatenate((tmp,np.array([i])),axis = 0)
		else:
			found.append(l)
			clusters[str(l)] = np.array([i])

	clusters_list =np.array([clusters[k] for k in clusters.keys()])
	N = len(clusters_list)

	# if mapping update cluster list
	if mapping is True:
		for i in range(N):
			for j in range(len(clusters_list[i])):
				clusters_list[i][j] = mapping_data[clusters_list[i][j]]
	# cluster list is now updated with correct atom names

	# add masses and positions
	clusters_pos,clusters_masses = np.empty(shape=(0,3)),np.empty([0])
	for cluster in clusters_list:
		position = np.zeros(3)
		m = 0
		for i in cluster:
			position += calphas.atoms[i].position
			m += calphas.atoms[i].mass

		position = np.array([position/float(len(cluster))])

		clusters_pos = np.concatenate((clusters_pos,position),axis = 0)
		clusters_masses = np.concatenate((clusters_masses,np.array([m])),axis = 0)


	return clusters_pos,clusters_masses,clusters_list




def nma_analysis(calphas,nodes_pos, masses, cutoff,mode = 0):
	'''
	Uses a modified version of ProDy where mass weighting of the hessian is
	done in buildHessian()

	takes as input to build an hessian and perform normal mode analysis:
	- coordinates of nodes to build geometric network
	- cutoff as edges threshold
	- masses to perform mass weighting in the hessian
	- calphas and nodes_pos are just to build the classes for the NMD file

	returns:
	- slowest mode eigenvector already divided in an array of eigenvectors
	for which entry i is the slowest mode eigenvector for atom i
	'''
	anm = prody.ANM("nd")
	anm.buildHessian(nodes_pos, cutoff = cutoff,masses = masses)
	anm.calcModes(3)

	slowest_mode = anm[mode]
	ev = slowest_mode.getEigvec().round(3)
	vectors = [[ev[i],ev[i+1],ev[i+2]] for i in range(0,len(ev),3) ]

	return np.asarray(vectors)



def graph_ground_truth(calphas, graph_cutoff):
	'''
	generates atomistc graph of carbons alphas
	'''

	n = len(calphas.atoms)
	G=nx.Graph()
	G.add_nodes_from(range(n))

	for i in range(n):
		for j in range(n):
			norm = np.linalg.norm(calphas.atoms[i].position - calphas.atoms[j].position)
			if norm <= graph_cutoff:
				G.add_edge(i,j)

	adj = nx.to_numpy_matrix(G)
	return np.squeeze(np.asarray(adj))

#
def graph_spectral_clustering(calphas, adjacency,n_clu,mapping_data = 0, mapping = False):
	'''
	apply spectral clustering using sklearn library.

	mapping arg is required when performing Laplacian Search to
	map indexes of subnetworks of randomly selected clusters
	to original atom indexes of the whole network
	'''

	sc = SpectralClustering(
		n_clusters = n_clu, 
		affinity='precomputed', 
		n_init=100, 
		assign_labels='discretize'
		)

	sc.fit_predict(adjacency)

	if mapping is True:
		clusters_pos, clusters_masses, cluster_list = ext_clusters_from_labels(sc.labels_,calphas,
														mapping_data = mapping_data, mapping= mapping )
	else:
		clusters_pos, clusters_masses, cluster_list = ext_clusters_from_labels(sc.labels_,calphas)
	return clusters_pos, clusters_masses, cluster_list


def rmsd(cluster_list, vectors, clu_vectors):
	'''
	not a proper rmsd - it is a 
	mean squared deviation of the CG modes wrt to AT modes,
	averaged along clusters
	'''

	N_clu = len(cluster_list)
	RMSD = 0
	N = len(vectors)
	for j in range(N_clu):
		clu_vector = clu_vectors[j]
		clu_true_vectors = vectors[cluster_list[j]]

		# normalize and multiply by 100
		if np.linalg.norm(clu_vector) < 1e-7: clu_vector +=  np.ones(3)*1e-6
		clu_vector = (clu_vector / np.linalg.norm(clu_vector))*100

		D = 0

		for v in clu_true_vectors:
			# normalize vectors to unit norm
			if np.linalg.norm(v) < 1e-7: v += np.ones(3)*1e-6

			v = (v / np.linalg.norm(v)) * 100
			d = minkowski(v,clu_vector,p = 2)
			D += d**2

		RMSD += float(D)
		
	return (RMSD/float(N_clu*N))/100.0

#
def rmsip(cluster_list,true_vectors, clu_vectors):
	true_modes = []
	clu_modes = []
	N_clu = len(cluster_list)
	for j in range(N_clu):
		clu_vector = clu_vectors[j]
		clu_true_vectors = true_vectors[cluster_list[j]]

		for k in range(len(clu_true_vectors)):
			v = clu_true_vectors[k]
			for w in range(3):
				true_modes.append(v[w])
				clu_modes.append(clu_vector[w])

	true_modes = np.asarray(true_modes)
	clu_modes = np.asarray(clu_modes)
	#rmsip_out = calcSubspaceOverlap(true_modes,clu_modes)

	true_modes *= 1 / (true_modes ** 2).sum(0) ** 0.5
	clu_modes *= 1 / (clu_modes ** 2).sum(0) ** 0.5
	overlap = np.dot(true_modes*100, clu_modes*100)
	#overlap = np.dot(true_modes.T, clu_modes)
	L = float(len(clu_modes))
	rmsip = np.sqrt(np.power(overlap, 2).sum() / (L*len(cluster_list)))
	return rmsip


def average_distance_from_neighbors(elements_pos, nn = 10):
	'''
	auxiliary function to calculate optimal cutoff for CG ENM
	'''

	n = len(elements_pos)
	if n < nn:
		nn = n
	nbrs = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(elements_pos)
	distances, indices = nbrs.kneighbors(elements_pos)
	avg_distances = np.mean(distances, axis = 1)
	# optimal cutoff will be conversionfactor * avg_clu_distances

	return np.mean(avg_distances)


#
# auxiliary functions (not interesting)
#

def together(one,two,three):
	new = []
	for element in one:
		new.append(np.array(element))
	for element in two:
		new.append(np.array(element))
	for element in three:
		new.append(np.array(element))
	return np.array([new[k] for k in range(len(new))])











