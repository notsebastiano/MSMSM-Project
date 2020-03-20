import numpy as np
import matplotlib.pyplot as plt
import prody
import MDAnalysis as mda

import matplotlib as mpl

import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from sklearn.cluster import SpectralClustering

import support_functions as sf
import laplacian_search as ls

#
# functions
#

def deviation(new_clu_list, vectors, clu_vectors):
	# auxiliary function
	r1 = sf.rmsd(new_clu_list, vectors, clu_vectors)
	r2 = sf.rmsd(new_clu_list, vectors, -1*clu_vectors)
	r = min([r1,r2])
	return r

#				 								#
# core functions of Laplacian Search algorithm  #
#				 								#

def bipartition(calphas, adj,clusters_pos, clusters_masses, cluster_list):
	# i get the original clusters as entry
	cp2, cm2, cl2 = [],[],[]
	# this will be my function, gets as entry and puts out new clu_list
	for cluster in cluster_list:
		
		sub_cluster_pos, sub_clusters_masses, sub_cluster_list = \
		sf.graph_spectral_clustering(
									 calphas, 
									 adj[cluster,:][:,cluster],
									 n_clu = 2 ,
									 mapping_data = cluster,
									 mapping = True
									 )

		cp2 = cp2 + [sub_cluster_pos[0],sub_cluster_pos[1]]
		cm2 = cm2 + [sub_clusters_masses[0],sub_clusters_masses[1]]
		cl2 = cl2 + [sub_cluster_list[0],sub_cluster_list[1]]

	cp2 = np.array([cp2[k] for k in range(len(cp2))])
	cm2 = np.array([cm2[k] for k in range(len(cm2))])
	cl2 = np.array([cl2[k] for k in range(len(cl2))])

	return cp2,cm2,cl2


def random_bipartition(adj,clusters_pos, clusters_masses, cluster_list,clu_vectors_old,
	r_old,
	threshold, alpha = 0.005, beta = 0.1):
	# start with existing clusters_pos, clusters_masses, cluster_list

	# random choice of one cluster
	index = np.random.choice(range(len(cluster_list)))
	X = cluster_list[index]

	if len(X) < 3:
		print "- cluster is too small - rejected"
		return clusters_pos, clusters_masses, cluster_list, r_old, clu_vectors_old

	else:
		# bipartition
		sub_cluster_pos, sub_clusters_masses, sub_cluster_list = \
		sf.graph_spectral_clustering(calphas, 
											adj[X,:][:,X],
											n_clu = 2 ,
											mapping_data = X,
											mapping = True)

		new_clu_list = sf.together(cluster_list[0:index],sub_cluster_list,cluster_list[index+1:])
		new_clu_pos = sf.together(clusters_pos[0:index],sub_cluster_pos,clusters_pos[index+1:])
		new_clu_masses = sf.together(clusters_masses[0:index],sub_clusters_masses,clusters_masses[index+1:])

		# evaluate increase in similarity
		avg_cd = sf.average_distance_from_neighbors(new_clu_pos,nn = 12)
		opt_cutoff = avg_cd/conversion_factor

		# new NMA of mass weighted clusters
		clu_vectors = sf.nma_analysis(new_clu_pos, new_clu_masses, 
									  cutoff = opt_cutoff,mode = 0)

		r = deviation(new_clu_list, vectors, clu_vectors)

		# accept if beyond a value
		delta = r_old - r

		# define cost function
		n1,n2,Nclu = len(sub_cluster_list[0]),len(sub_cluster_list[1]),len(clu_vectors)

		gain = delta - alpha*((1.0/n1)+(1.0/n2)) - beta*Nclu

		accept = False
		if gain > threshold: accept = True

		# return state
		if accept == True:
			print "accepted:",gain
			return new_clu_pos, new_clu_masses, new_clu_list, r,clu_vectors
		else:
			# return previous state
			print "rejected:",gain
			return clusters_pos, clusters_masses, cluster_list, r_old,clu_vectors_old




#########################################	

#
# main: setting values
# execution of the clustering algorithm
#

name = "3kin"

n_bipartitions = 2
iterations = 3000

threshold = 0.010
alpha = 0.00005
beta = 0.001

mode = 0

# setting the universe and the ground truth network
u = mda.Universe('proteins/{}.pdb'.format(name))
calphas = u.select_atoms('protein and name CA')

#coords
calphas_pos = np.asarray([atom.position for atom in calphas])
n = len(calphas_pos)
masses = np.asarray([calphas.atoms[i].mass for i in range(n)])

# setting values
truth_cutoff = 13.0
avg_d = sf.average_distance_from_neighbors(calphas_pos,nn = 12)
conversion_factor = avg_d/truth_cutoff

# ground truth slowest mode vectors (should check for other modes too TO GET A BETTER FIT)
vectors = sf.nma_analysis(calphas_pos, masses, cutoff = truth_cutoff,mode = 0)
adj = sf.graph_ground_truth(calphas,graph_cutoff=truth_cutoff)


if 1:
	print "starting bipartitions"
	clusters_pos, clusters_masses, cluster_list = sf.graph_spectral_clustering(calphas, adj, n_clu = 2 )

	t = 1
	while t <= n_bipartitions:
		clusters_pos, clusters_masses, cluster_list = bipartition(calphas,adj,clusters_pos, clusters_masses, cluster_list)
		t = t+1
	print "number of clusters obtained as starting state: {}".format(str(len(cluster_list)))

	# evaluation of r_old
	avg_cd = sf.average_distance_from_neighbors(clusters_pos,nn = 9)
	opt_cutoff = avg_cd/conversion_factor

	# new NMA of mass weighted clusters
	clu_vectors = sf.nma_analysis(clusters_pos, clusters_masses, 
								  cutoff = opt_cutoff,mode = 0) #18
	r = deviation(cluster_list, vectors, clu_vectors)

	T = 0
	R,clu_lengths = [],[]

	while T < iterations:
		# random bipartitions
		clusters_pos, clusters_masses, cluster_list,r,clu_vectors = \
		random_bipartition(adj,clusters_pos, clusters_masses, cluster_list,clu_vectors,
						   r_old = r,
						   threshold = threshold,
						   alpha = alpha,
						   beta = beta)
		R.append(r)
		clu_lengths.append(len(clusters_pos)/float(n))

		k = len(np.array(clusters_masses).flatten())
		print "r now is: {} and cluster are {} and atoms are {}".format(str(r),str(len(clusters_pos)),str(k))

		T = T+1

  	#
  	# saving data and creating nmd file for visualization of modes
  	#

	data_path = "{}/data/".format(str(name))
	# creating and saving nmd file
	CG_modes = clu_vectors.flatten()
	anm1 = prody.NMA("CG_modes_search")
	anm1.addEigenpair(CG_modes)
			
	filename = data_path +  "CG_NMD.nmd"
	clus = atoms(clusters_pos)
	prody.writeNMD(filename, anm1, clus)

	# saving data
	np.savetxt(data_path +"distances_and_Nclu.csv", np.array([R,clu_lengths]))
	np.savetxt(data_path + "clu_pos.csv", np.array(clusters_pos))
	np.savetxt(data_path + "clu_masses.csv", np.array(clusters_masses))
	np.savetxt(data_path + "clu_vectors.csv", np.array(clu_vectors))
	# creo file csv con lista di atomi in cluster - un fil per ogni cluster
	for i in range(len(cluster_list)):
		np.savetxt(data_path +"clulist/"+ "clu_list{}.csv".format(str(i)), np.array(cluster_list[i]))

	#
	# code for figures and further analysis is not uploaded 
	#




