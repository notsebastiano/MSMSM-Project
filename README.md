# MSMSM-Project
Project for the course of Multi Scale Methods in Soft Matter - University of Trento
held by professor R.Potestio

Thus project proposes a mapping strategy that ensures the reproducibility of slow modes in a Normal Mode Analysis of the system (a protein, in this case) studied through the paradigm of Elastic Network Models. Laplacian clustering of the ENM graph yields Coarse Grained sites. These sites can be used to model a new elastic network of supernodes that is still able to reproduce the same slow modes, compatible with collective motion and conformational changes, eventually identifying relevant domains in conformational changes.

The project is divided into two main sections: the first applies sklearn laplacian clustering to identify clusters and uses an embedding in k dimensions to apply k-means. The second is an iterative bipartition of the system by selecting randomly the clusters and proposing new configurations. The idea is to create clusters from successive bipartitions of the network,
where a division of a cluster occurs if this guarantees an improvement in the reproduction of the normal modes, without however creating too small clusters
