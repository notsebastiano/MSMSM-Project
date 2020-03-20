import numpy as np
import matplotlib.pyplot as plt
import prody
import MDAnalysis as mda

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import SpectralClustering
from sklearn import metrics

import support_functions as sf

#
# functions
#

def spectral_automated(calphas, adj, vectors, n_clu, conversion_factor,MODE):

	# get clusters (pos,masses,list) from sklearn spectral clustering
	clusters_pos, clusters_masses, cluster_list = sf.graph_spectral_clustering(calphas, adj, n_clu )
	avg_cd = sf.average_distance_from_neighbors(clusters_pos,nn = 12)
	opt_cutoff = avg_cd/conversion_factor

	# new NMA of mass weighted clusters
	clu_vectors = sf.nma_analysis(calphas,clusters_pos, clusters_masses, cutoff = opt_cutoff,mode = MODE,save = False) #18

	# COMPARISON OF AVG VECTORS AND CLUSTERS NMA
	r1 = sf.rmsd(cluster_list, vectors, clu_vectors)
	r2 = sf.rmsd(cluster_list, vectors, -1*clu_vectors)
	r = min([r1,r2])

	return r, cluster_list, clu_vectors


def protein_analysis(PDB,MODE):

	u = mda.Universe(PDB)
	calphas = u.select_atoms('name CA')

	#coords
	calphas_pos = np.asarray([atom.position for atom in calphas])
	n = len(calphas_pos)
	masses = np.asarray([calphas.atoms[i].mass for i in range(n)])

	# setting values and
	# create range of cluster numbers which is denser in region of transition
	truth_cutoff = 11.0 # 11.0 is ideal
	clu_range =np.concatenate(
		(np.linspace(int(n-n/2.0),int(n*0.2),num = 10),
		np.linspace(int(n*0.2)-5,10,num = 20)),
		axis = 0)

	avg_d = sf.average_distance_from_neighbors(calphas_pos,nn = 12)
	conversion_factor = avg_d/truth_cutoff

	# ground truth
	vectors = sf.nma_analysis(calphas,calphas_pos, masses, cutoff = truth_cutoff,mode = MODE)
	adj = sf.graph_ground_truth(calphas,graph_cutoff=truth_cutoff)

	distances = []
	# one mode (eigenvector) saved for every value in clu range
	for n_clu in clu_range:
		r,clu_list,clu_vectors = spectral_automated(calphas, adj, vectors, int(n_clu), conversion_factor,MODE)
		distances.append(r)

	return clu_range/float(n),np.array(distances)

#
# actual analysis #
#

if 1:
	NAMES = ["3kin","3l1c","3j6h"]

	mode_selected = 2
	print "starting statistical analysis of mode {}".format(str(mode_selected))

	for name in NAMES:	
		j,iterations = 0,5 #iteration retained just 5
		save_path = "{}/spectral_iterations_mode{}/".format(name,str(mode_selected))

		while j < iterations:
			clu_range,distances,truth_mode,CG_modes = protein_analysis("proteins/{}.pdb".format(name),MODE = mode_selected)
			np.savetxt(save_path + name +"_distances_iteration_{}.csv".format(str(j)), distances)
			j = j+1

		np.savetxt(save_path + name +"_CLU_RANGE.csv", clu_range) #clu range is constant in every iteration












