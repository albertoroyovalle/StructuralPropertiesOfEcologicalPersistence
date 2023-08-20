import numpy as np
from scipy.stats import linregress
from collections import namedtuple
import random
import networkx as nx

# Decompositions (in-silico extinction experiments)
###################################################

def Decomposition_Bipartite(M, nbsimu, independent = False):
    """
    Simulates the network decompositions (in-silico extinction experiments)

    Parameters
    ----------
    mat : numpy array of shape (S,S), S being the number of species
        Adjacency matrix of the network.
        
    nbsimu : integer
        Number of simulations (decompositions) to perform.
            
    independent : bool
        Should the species having no incoming links be considered as 
        independent (i.e. not undergo secondary extinction)?

    Returns
    -------
    S: integer
        Number of species in the network  
        
    L: integer
        Number of links (edges) in the network (excluding cannibalism)
        
    b: float
        Shape of the L~S relationship defined as log(L)/log(S/2)
        
    z: float
        Proportion of basal species
        
    sseq: numpy array of shape (nbsimu, S)
        Number of species along each decomposition 
        (one row is one whole decomposition of the network)
   
    lseq: numpy array of shape (nbsimu, S)
        Number of links along each decomposition 
        (one row is one whole decomposition of the network)

    """

    mat = M
    #Species number
    P,A = mat.shape
    S=P+A
    #link number
    entryList = M.flatten()
    linkList = entryList[entryList!=0]
    linkNum = len(linkList)
    E = linkNum
    L=E

    z = sum(np.sum(mat, axis=1)==0) # Number of independent species
    b = np.log(L)/np.log(0.5*S) # Shape of the L~S relationship

    #Definition of the SSEQ and LSEQ
    sseq=np.zeros((nbsimu,P+A))
    lseq=np.zeros((nbsimu,P+A))


    #percolation loop
    SList = []
    for i in range(nbsimu):
        iter=0
        mat = M
        #Species number
        P,A = mat.shape
        #link number
        entryList = mat.flatten()
        linkList = entryList[entryList!=0]
        linkNum = len(linkList)
        E = linkNum

        #First column SSEQ and LSEQ
        sseq[i][iter]=P+A
        lseq[i][iter]=E

        n = 0
        while E != 0:
            iter=iter+1
            S = P+A
            SList.append(S)
            
            #select species
            k = np.random.randint(0,2)
            deathSpecies = np.random.randint(0,[P,A][k])
            
            #direct extinction
            mat = np.delete(mat,deathSpecies,axis=k)
        
            
            #indirect extiction
            indirectExtCol = ~np.all(mat==0,axis=0)
            mat = mat[:,indirectExtCol]
            indirectExtRow = ~np.all(mat==0,axis=1)
            mat = mat[indirectExtRow,:]
            
            
            #Species number
            P,A = mat.shape
            #link number
            entryList = mat.flatten()
            linkList = entryList[entryList!=0]
            linkNum = len(linkList)
            E = linkNum

            sseq[i][iter]=P+A
            lseq[i][iter]=E

    mat = M
    P,A = mat.shape
    S=P+A

    result = namedtuple('experes', ('S', 'L', 'b', 'z', 'sseq', 'lseq'))
    if not independent :
        z = 0
    return(result(S, L, b, z/S, sseq, lseq))





def ExtinctionExperimentNetworkx(mat, nbsimu, independent = False):
    """
    Simulates the network decompositions (in-silico extinction experiments)

    Parameters
    ----------
    mat : numpy array of shape (S,S), S being the number of species
        Adjacency matrix of the network.
        
    nbsimu : integer
        Number of simulations (decompositions) to perform.
            
    independent : bool
        Should the species having no incoming links be considered as 
        independent (i.e. not undergo secondary extinction)?

    Returns
    -------
    S: integer
        Number of species in the network  
        
    L: integer
        Number of links (edges) in the network (excluding cannibalism)
        
    b: float
        Shape of the L~S relationship defined as log(L)/log(S/2)
        
    z: float
        Proportion of basal species
        
    sseq: numpy array of shape (nbsimu, S)
        Number of species along each decomposition 
        (one row is one whole decomposition of the network)
   
    lseq: numpy array of shape (nbsimu, S)
        Number of links along each decomposition 
        (one row is one whole decomposition of the network)

    """
    
    #-------- Graph initialization -------- 
    G = nx.from_numpy_array(mat)

    #-------- Network structure -------- 
    S=  G.number_of_nodes()
    L=  G.number_of_edges()
    z = sum(np.sum(mat, axis=1)==0) # Number of independent species
    b = np.log(L)/np.log(0.5*S) # Shape of the L~S relationship

    # Number of species, initialization of the SSEQ and LSEQ matrix
    sseq=np.ones([nbsimu,G.number_of_nodes()+1],dtype=int)*G.number_of_nodes()  # All decompositions start with all species   (Creation of nbsimu matrices S)
    # Number of links
    lseq=np.ones([nbsimu,G.number_of_nodes()+1],dtype=int)*G.number_of_edges()  # All decompositions start with all links   (Creation of nbsimu matrices L)


    iter=0
    #-------- Iterative decompositions -------- nbsimu simulations
    for i in range(nbsimu):
        while G.number_of_edges() != 0: #While there are links we continue the decomposition.
            iter=iter+1
            #Remove a random node and isolated nodes
            node=random.choice(np.array(G.nodes()))
            G.remove_node(node)
            G.remove_nodes_from(list(nx.isolates(G)))  #Isolated nodes are removed
            
            #Update the matrices
            sseq[i][iter]=G.number_of_nodes()
            lseq[i][iter]=G.number_of_edges()
            
            if G.number_of_edges()==0:
                sseq[i][iter:]=0
                lseq[i][iter:]=0
        
        iter=0
        G = nx.from_numpy_array(mat)        #Reinitialization for the next loop
        

    result = namedtuple('experes', ('S', 'L', 'b', 'z', 'sseq', 'lseq'))
    if not independent :
        z = 0
    return(result(S, L, b, z/S, sseq, lseq))


def ExtinctionExperimentNetworkxV2(mat, nbsimu, independent = False):
    """
    Simulates the network decompositions (in-silico extinction experiments)

    Parameters
    ----------
    mat : numpy array of shape (S,S), S being the number of species
        Adjacency matrix of the network.
        
    nbsimu : integer
        Number of simulations (decompositions) to perform.
            
    independent : bool
        Should the species having no incoming links be considered as 
        independent (i.e. not undergo secondary extinction)?

    Returns
    -------
    S: integer
        Number of species in the network  
        
    L: integer
        Number of links (edges) in the network (excluding cannibalism)
        
    b: float
        Shape of the L~S relationship defined as log(L)/log(S/2)
        
    z: float
        Proportion of basal species
        
    sseq: numpy array of shape (nbsimu, S)
        Number of species along each decomposition 
        (one row is one whole decomposition of the network)
   
    lseq: numpy array of shape (nbsimu, S)
        Number of links along each decomposition 
        (one row is one whole decomposition of the network)

    """
    #-------- Network structure -------- 
    #-------- Graph initialization -------- 
    G = nx.from_numpy_array(mat)
    S=  G.number_of_nodes()
    L=  G.number_of_edges()
    z = sum(np.sum(mat, axis=1)==0) # Number of independent species

    b = np.log(L)/np.log(0.5*S) # Shape of the L~S relationship

    # Number of species, initialization of the SSEQ and LSEQ
    sseq=np.ones([nbsimu,G.number_of_nodes()+1],dtype=int)*G.number_of_nodes()  # All decompositions start with all species   (Creation of nbsimu matrices S)
    # Number of links
    lseq=np.ones([nbsimu,G.number_of_nodes()+1],dtype=int)*G.number_of_edges()  # All decompositions start with all links   (Creation of nbsimu matrices L)


    iter=0
    #-------- Iterative decompositions -------- nbsimu simulations
    for i in range(nbsimu):
        while G.number_of_edges() != 0: #While there are links we continue the decomposition.
            iter=iter+1

            #-------Remove a random node 
            node=random.choice(np.array(G.nodes()))
            G.remove_node(node)

            # Remove isolated nodes and no-income nodes (in case of directed graph).
            ##################################################################################################################################################################
            try: #Intento detectar si hay nodos sin entradas pero con salidas... En un grafo simple esto no tiene sentido, pero en un grafo dirigido si.
                mydict=G.pred
                no_income_nodes = [k for k, v in G.pred.items() if v == {}]
            except:
                no_income_nodes = []
            #SI UTILIZAMOS EL GRAFO SIMPLE IGNORAR TODO ESTO DEL TRY EXCEPT.

            while (list(nx.isolates(G))!=[] or list(no_income_nodes)!=[]):  #Siempre que existan nodos aislados y/o sin entradas(en caso de directed graph), los eliminamos.
                G.remove_nodes_from(list(nx.isolates(G)))  #Isolated nodes are removed

                try:
                    mydict=G.pred
                    no_income_nodes = [k for k, v in G.pred.items() if v == {}]
                    G.remove_nodes_from(list(no_income_nodes))
                except:
                    no_income_nodes = []
            ##################################################################################################################################################################
                    
                
                
            
            #Update the matrices
            sseq[i][iter]=G.number_of_nodes()
            lseq[i][iter]=G.number_of_edges()
            
            if G.number_of_edges()==0:
                sseq[i][iter:]=0
                lseq[i][iter:]=0
        
        iter=0
        G = nx.from_numpy_array(mat)        #Reinitialization for the next loop
        

    result = namedtuple('experes', ('S', 'L', 'b', 'z', 'sseq', 'lseq'))
    if not independent :
        z = 0
    return(result(S, L, b, z/S, sseq, lseq))