import itertools
import pandas as pd
import scipy.stats as stats
import numpy as np
import networkx as nx

MDBs = ['H2','CH2','CO2','O','CHOH','NCH','O(N-H-)','S','CONH','PO3H','NH3(O-)','SO3','CO', 'C2H2O', 'H2O']

def HD_sMDiN_analysis(sampr):
    
    samp, net = sampr
    sample = {}
    sample['Name'] = samp
    #print(samp)
    sample['Deg'] = dict(net.degree())
    sample['Betw'] = nx.betweenness_centrality(net)
    sample['Closeness'] = nx.closeness_centrality(net)

    # MDB Impact
    sample['MDB_Impact'] = dict.fromkeys(MDBs, 0) # MDBs from the transformation list
    for i in net.edges():
        sample['MDB_Impact'][net.edges()[i]['Transformation']] = sample['MDB_Impact'][
            net.edges()[i]['Transformation']] + 1

    # GCD-11
    # Corr_Mat
    orbits_t = calculating_orbits(net) # Calculating orbit number for each node
    orbits_df = pd.DataFrame.from_dict(orbits_t).T # Transforming into a dataframe

    # Signature matrices
    corrMat_ar = stats.spearmanr(orbits_df)[0] # Calculating spearman correlation to obtain 11x11 signature of the network - GCM
    corrMat_tri = np.triu(corrMat_ar) # Both parts of the matrix are equal, so reducing the info to the upper triangle

    # Pulling the signature orbit n (u) - orbit m (v) correlations from the upper triangular matrix of the GCM
    # Making the signature of the sample MDiN into a column of the dataset
    samp_col = {}
    orbits = [0,1,2,4,5,6,7,8,9,10,11]
    #print(corrMat_tri)
    for u in range(len(corrMat_tri)):
        for v in range(u+1, len(corrMat_tri)):
            samp_col[str(orbits[u]) + '-' + str(orbits[v])] = corrMat_tri[u,v]
        sample['GCD'] = samp_col
    return sample


def calculating_orbits(GG):
    """Calculates the number of times each node of the network is in each possible (non-redundant) orbit in graphlets (maximum
    4 nodes).
    
    Function is not very efficient, all nodes are passed, every graphlet is 'made' for each node present in it so it is made
    multiple times.
    
       GG: networkx graph;
    
       returns: dict; dictionary (keys are the nodes) of dictionaries (keys are the orbits and values are the number of times)
    """
    
    node_orbits = {} # To store results

    for i in GG.nodes():

        node_orbits[i] = {} # To store results
        orbits = node_orbits[i]

        # 2 node graphlets - orbit 0
        orbits['0'] = GG.degree(i)

        # 3 node graphlets - orbit 1,2 (and 3 redundant)
        node_neigh = list(GG.neighbors(i))

        # orbit 1 and 4 and 6 and 8 and 9
        n_orb = 0
        n_orb4 = 0
        n_orb6 = 0
        n_orb8 = 0
        n_orb9 = 0

        # orbit 1
        for j in node_neigh:
            neigh_neigh = list(GG.neighbors(j)) # Neighbours of the neighbour j of i
            neigh_neigh.remove(i) # Remove i since i is a neighbour of j
            for common in nx.common_neighbors(GG, i, j):
                neigh_neigh.remove(common) # Remove common neighbours of i and j
            n_orb = n_orb + len(neigh_neigh)


            # orbit 4 and 8
            for n3 in neigh_neigh:
                neigh_neigh_neigh = list(GG.neighbors(n3)) # Neighbours of the neighbour n3 of the neighbour j of i
                #neigh_neigh_neigh.remove(j)
                #if i in neigh_neigh_neigh:
                    #neigh_neigh_neigh.remove(i)     
                for common in nx.common_neighbors(GG, j, n3):
                    if common in neigh_neigh_neigh:
                        neigh_neigh_neigh.remove(common)

                for common in nx.common_neighbors(GG, i, n3):
                    if common in neigh_neigh_neigh:
                        neigh_neigh_neigh.remove(common)
                        # orbit 8
                        if common != j:
                            #print(i,j,n3,common)
                            n_orb8 = n_orb8 + 1/2 # always goes in 2 directions so it will always pass like this

                n_orb4 = n_orb4 + len(neigh_neigh_neigh)
                # print(neigh_neigh_neigh)

            # orbit 6 and 9
            for u,v in itertools.combinations(neigh_neigh, 2):
                if not GG.has_edge(u,v):
                    n_orb6 = n_orb6 + 1
                else:
                    n_orb9 = n_orb9 + 1         

        orbits['1'] = n_orb

        # orbit 2 and 5
        n_orb = 0
        n_orb5 = 0
        for u,v in itertools.combinations(node_neigh, 2):
            if not GG.has_edge(u,v):
                n_orb = n_orb + 1

                # orbit 5
                neigh_u = list(GG.neighbors(u))
                neigh_u.remove(i)
                for common in nx.common_neighbors(GG, i, u):
                    neigh_u.remove(common)

                neigh_v = list(GG.neighbors(v))
                neigh_v.remove(i)
                for common in nx.common_neighbors(GG, i, v):
                    neigh_v.remove(common)

                for common in nx.common_neighbors(GG, v, u):
                    if common in neigh_u:
                        neigh_u.remove(common)
                    if common in neigh_v:
                        neigh_v.remove(common) 

                n_orb5 = n_orb5 + len(neigh_u)
                n_orb5 = n_orb5 + len(neigh_v)

        orbits['2'] = n_orb

        # 4 node graphlets - orbit 4,5,6,7,8,9,10,11 (and 12,13,14 redundant)

        # orbit 4
        orbits['4'] = n_orb4

        # orbit 5
        orbits['5'] = n_orb5

        # orbit 6
        orbits['6'] = n_orb6

        # orbit 7 and 11
        n_orb = 0
        n_orb11 = 0
        for u,v,j in itertools.combinations(node_neigh, 3):
            n_edge = [GG.has_edge(a,b) for a,b in itertools.combinations((u,v,j), 2)]
            #print(sum(n_edge))
            if sum(n_edge) == 0:
                n_orb = n_orb + 1
            elif sum(n_edge) == 1:
                n_orb11 = n_orb11 + 1

        orbits['7'] = n_orb

        # orbit 8
        orbits['8'] = int(n_orb8)

        # orbit 9
        orbits['9'] = n_orb9

        # orbit 10
        n_orb = 0
        for j in node_neigh:
            neigh_neigh = list(GG.neighbors(j))
            neigh_neigh.remove(i)
            for u,v in itertools.combinations(neigh_neigh, 2):
                if sum((GG.has_edge(i,u), GG.has_edge(i,v))) == 1:
                    if not GG.has_edge(u,v):
                        n_orb = n_orb + 1

        orbits['10'] = n_orb

        # orbit 11
        orbits['11'] = n_orb11
    
    return node_orbits