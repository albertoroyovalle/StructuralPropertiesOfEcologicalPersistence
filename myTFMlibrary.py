import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import decomposition
import robustness
import timeit
import networkx as nx
from scipy.special import gamma


def erdos_renyi(N, p):
    # Erdos-Renyi random graph creator using network
    # INPUT
    # - N (int) matrix size
    # - p (float) probability of connection
    # OUTPUT
    # - A (numpy array) adjacency matrix

    G = nx.erdos_renyi_graph(N, p)
    # To numpy adjacency matrix
    A = nx.adjacency_matrix(G)
    A = A.todense()
    return A

def xiConnTheoricalGamma(Xi_range):
    # Theorical connectance for a given nestedness
    # INPUT
    # - Xi_range (float) nestedness
    # OUTPUT
    # - connectance (float) matrix connectance
    connectance =  (Xi_range * gamma(Xi_range)**2) / (2 * gamma(2*Xi_range))
    return(connectance)


def xiConnRelationship(N, xi):
    """
    INPUT
    - N (int) matrix size
    - xi (float) nested profile
    OUTPUT
    - C (float) matrix connectance
    """
    E = 0  # edge counter
    # column loop
    for i in range(N):
        x = i/N  # tessellate
        # unite ball equation
        y = 1-(1-x**(1/xi))**xi
        # row loop
        for j in range(N):
            if j/N >= y:
                E += 1
    C = round(E/(N*N), 2)
    return(C)

def stab_carpentier(mat, inttype, mu = 0, sigma2 = 1, d = 0, nbsimu = 100):
    # Calculate the stability of a matrix based on Carpentier's work
    # INPUT
    # - mat (numpy array) adjacency matrix
    # - inttype (string) type of interaction
    # - mu (float) mean of the interaction strength
    # - sigma2 (float) variance of the interaction strength
    # - d (float) probability of extinction
    # - nbsimu (int) number of simulations
    # OUTPUT
    # - eigenvalue (float) stability of the matrix

    A = mat!=0 # Adjacency matrix A
    W = np.random.normal(mu,sigma2,
                         size=(nbsimu, *A.shape)) # Interaction strength matrix
    M = W*A[np.newaxis,:,:] # Community matrix M
       
    if inttype == "mutualistic":
        M = abs(M) # All values are positive (half-normal distribution)

     #### Observed eigenvalues ####
            
    eigv,eigvect = np.linalg.eig(M) # Eigenvalues and eigenvectors

    eig = np.amax(eigv.real, 1) # Maximal real part of the eigenvalues
    return eig.mean()

def stability_check(jacobian_matrix):
    eigenvalues = np.linalg.eigvals(jacobian_matrix)
    real_parts = np.real(eigenvalues)
    indx = np.array(abs(eigenvalues)).argmax()
    return real_parts[indx]


def find_max_real_eigenvalue(A):
    eigenvalues, _ = np.linalg.eig(A)
    max_eigenvalue = max(eigenvalues, key=lambda x: abs(x.real))
    return max_eigenvalue.real


def calculate_lambda_max(matrix):
    eigenvalues = np.linalg.eigvals(matrix)
    lambda_star = -np.max(np.real(eigenvalues))
    return lambda_star.real


def find_max_real_abs_eigenvalue(A):
    eigenvalues, _ = np.linalg.eig(A)
    max_eigenvalue = max(eigenvalues, key=lambda x: abs(x.real))
    return abs(max_eigenvalue)


def from_biadjacency_matrix(M):
    b = M
    r, s = b.shape
    mat = np.vstack((np.hstack((np.zeros((r, r)), b)),
                    np.hstack((b.T, np.zeros((s, s))))))
    return(mat)


def NestednessCreator(N, xi, P):
    xList = np.arange(0, N)/N  # x values
    # initialize
    yList = []  # y list
    curveDict = dict({})  # x-y dict

    # column loop
    for x in xList:

        # nested profile
        y = 1-(1-x**(1/xi))**xi
        yList.append(y)
        curveDict[round(x, 2)] = y
    # BUILD PERFECT NESTED MATRIX
    M = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):

            iNorm = round(i/N, 2)
            jNorm = j/N

            curvePoint = curveDict[iNorm]

            if curvePoint <= jNorm:
                M[j, i] = 1

    # BUILD NOISE NESTED MATRIX
    linkList = M.nonzero()
    linkRow = linkList[0]
    linkCol = linkList[1]

    RemovedIndex = []

    # link loop
    for l in range(len(linkRow)):

        p = np.random.uniform(0, 1)

        if p < P:
            RemovedIndex.append(l)
    for index in RemovedIndex:
        M[linkRow[index], linkCol[index]] = 0

    zeroEntries = np.where(M == 0)  # Look for zero entries
    zeroRow = zeroEntries[0]
    zeroCol = zeroEntries[1]

    ZeroIndex = range(len(zeroRow))

    FillIndex = random.sample(ZeroIndex, len(RemovedIndex))

    for i in FillIndex:
        M[zeroRow[i], zeroCol[i]] = 1
    return(np.rot90(M, k=3).transpose())


def NestednessCreatorSymmetric(N, xi, P):
    xList = np.arange(0, N)/N  # x values
    # initialize
    yList = []  # y list
    curveDict = dict({})  # x-y dict

    # column loop
    for x in xList:

        # nested profile
        y = 1-(1-x**(1/xi))**xi
        yList.append(y)
        curveDict[round(x, 2)] = y
    # BUILD PERFECT NESTED MATRIX
    M = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):

            iNorm = round(i/N, 2)
            jNorm = j/N

            curvePoint = curveDict[iNorm]

            if curvePoint <= jNorm:
                M[j, i] = 1
    M = np.rot90(M, k=3)
    M = np.triu(M, k=1).transpose()+np.triu(M)

    # BUILD NOISE NESTED MATRIX
    linkList = M.nonzero()
    linkRow = linkList[0]
    linkCol = linkList[1]

    RemovedIndex = []

    # link loop
    for l in range(len(linkRow)):

        p = np.random.uniform(0, 1)

        if p < P:
            RemovedIndex.append(l)
    for index in RemovedIndex:
        M[linkRow[index], linkCol[index]] = 0

    zeroEntries = np.where(M == 0)  # Look for zero entries
    zeroRow = zeroEntries[0]
    zeroCol = zeroEntries[1]

    ZeroIndex = range(len(zeroRow))

    FillIndex = random.sample(ZeroIndex, len(RemovedIndex))

    for i in FillIndex:
        M[zeroRow[i], zeroCol[i]] = 1

    return(M)


def ConnectanceRandomMatrix(n, C):
    # Function that creates an square matrix of zeros with a density of ones (connectance) in random postions.
    # Inputs
    # n= dimention of the matrix
    # C connectance. Number in the interval [0 , 1]
    # Some problems. Sometiemes the number of elemets different form 0---- Links=connectance*dim^2 its is a non int number...This should be fixed in some way
    indexes = []
    for i in range(n):
        for j in range(n):
            indexes.append([i, j])
    L = int(C*(n*n))
    UpdatedList = random.sample(indexes, L)

    M = np.zeros((n, n), dtype=int)
    for i in UpdatedList:
        row = i[0]
        column = i[1]

        # Modification of the element  M[row][column]=1
        M[row][column] = 1

    M = np.tril(M, k=-1).transpose()+np.tril(M, k=-1)

    return(M)


def symmetricBinaryMatrix(S, C):
    """
    Generate a symmetric bynary random matrix

    INPUT
    - S (int): matrix size
    - C (float): connectance

    OUTPUT
    - M (2d array): binary matrix

    """

    M = np.zeros((S, S), dtype=int)

    for i in range(S):
        for j in range(i):

            p = np.random.uniform(0, 1)

            if p < C:
                M[i, j] = 1
                M[j, i] = 1
    M = np.tril(M, k=-1).transpose()+np.tril(M)
    return(M)


def BinaryMatrixGenerator(S, C):
    """
    Binary matrix with fixed size and connectance;

    INPUT
    - S (float): species number;
    - C (float): connectance;

    OUTPUT
    - M (SxS array): binary matrix
    """

    boolMat = np.random.uniform(0, 1, size=(S, S))
    M = (boolMat < C).astype(int)
    # tr_diag=np.tril(M)
    # tr=np.tril(M,k=-1)
    # trans=tr.transpose()
    M = np.tril(M, k=-1).transpose()+np.tril(M)
    return(M)


def NumberOfEdges(mat):
    A = mat.flatten()
    A = A[A != 0]
    L = len(A)
    return(L)


def createMosaic():
    # Horizontally stacked subplots
    fig2, ax = plt.subplots(2, 5, figsize=(30, 20))
    ax1 = ax.flatten()

    return(fig2, ax1)


def Figura2aCreator(dim, connectance, removed, lost, S, deltaS, path_figures):
    # Figure 2a (equivalent)
    ##########################
    fig = plt.figure()
    #### Unique combinaison in the lost~removed plane ####
    dots, likely = np.unique(tuple([removed.flatten(),
                                    lost.flatten()]),
                             axis=1, return_counts=True)

    #### Dots with size proportional to their likelihood in the lost~removed plane ####
    plt.scatter(*dots, s=likely/100,  c="grey", label="Observed")

    #### Predictions of the lost~removed relationship ####
    plt.plot(np.linspace(0, S, S*10), deltaS*np.linspace(0, S, S*10),
             c="black", label="Predicted")
    plt.xlabel("Number of species removed (r)")
    plt.ylabel("Number of species lost (n)")
    plt.axhline(0.5*S, color="black",
                linestyle="dotted")  # 50% of species lost
    plt.legend()
    # plt.show()
    plt.savefig(path_figures+"Extinctions_" +
                str(dim)+"_"+str(connectance)+".pdf")
    plt.close()


def Figura3aCreator(dim, connectance, S, sseq, b, z, path_figures):
    # Extended Data Figure 3a
    ##########################
    fig = plt.figure()
    xs = np.round(np.linspace(0.01, 1, S), 2)  # Various robustness threshold
    robs = []
    for x in xs:  # For each threshold
        robs.append([*robustness.robx(sseq, S, b, z, x)])  # Compute robustness
    robs = np.array(robs)
    plt.errorbar(xs, robs[:, 0],
                 yerr=robs[:, 1]**0.5/2,
                 c="black", fmt='o', label="Observed")
    plt.plot(xs, robs[:, 2], c="black", label="Predicted", zorder=-1)
    plt.xlabel("x")
    plt.ylabel("Robustness at threshold x")
    plt.legend()
    plt.savefig(path_figures+"Robustness_" +
                str(dim)+"_"+str(connectance)+".pdf")
    # Computing rob_obs for multiple networks allows to obtain
    # Extended Data Figure 3b.
    plt.close()


def TablaRobustnessCreator(dim, connectance, nombretabla, rob_pred, rob_obs):
    # Datasave Module
    # Cambiar cuando se pase a linux.
    path_output = "/home/vant/Documentos/TFM/codes/Figuras/"
    try:
        ReadTabla = pd.read_csv(path_output+nombretabla+".csv")
        print(nombretabla+".csv exist")

        df = pd.DataFrame({"Dimention": [dim], "Connectance": [
                          connectance], "Predicted_Rob": [rob_pred], "Observed_Rob": [rob_obs]})
        ReadTabla = pd.concat([ReadTabla, df])
        ReadTabla.to_csv(path_output+nombretabla+".csv", index=False)
        print(nombretabla+".csv updated")

    except Exception:
        RobustnessDataframe = pd.DataFrame({"Dimention": [dim], "Connectance": [
                                           connectance], "Predicted_Rob": [rob_pred], "Observed_Rob": [rob_obs]})
        RobustnessDataframe.to_csv(path_output+nombretabla+".csv", index=False)
        print("Robustness CSV has been created in you working directory")


def MosaicPlotCreator(dim, connectance, removed, lost, sseq, lseq, b, S, ax1, deltaS, z):
    dots, likely = np.unique(tuple([removed.flatten(),
                                    lost.flatten()]),
                             axis=1, return_counts=True)
    breg, areg, r2reg, r2b = decomposition.R2L(sseq, lseq, b)
    deltaS_new, r2lost_new, lost_new, removed_new = robustness.R2deltaS(
        sseq, S, breg, z)
    dots_new, likely_new = np.unique(tuple([removed_new.flatten(),
                                            lost_new.flatten()]),
                                     axis=1, return_counts=True)
    # Dot Media module:
    y = []
    x = []
    for i in range(dim):
        index = np.where(dots[0] == i)
        y.append(
            np.mean(np.multiply(dots[1][index], likely[index])) / np.mean(likely[index]))
        x.append(i)

    ax1.scatter(*dots, s=likely/100,  c="grey", label="Observed probs.")

    #### Dots with size proportional to their likelihood in the lost~removed plane ####

    #### Predictions of the lost~removed relationship ####
    ax1.plot(np.linspace(0, S, S*10), deltaS*np.linspace(0, S, S*10),
             c="black", label="Predicted")

    ax1.plot(np.linspace(0, S, S*10), deltaS_new*np.linspace(0, S, S*10),
             c="green", label="Experimental")
    ax1.plot(x, y, c="blue", label="Weighted mean observed")

    ax1.set(xlabel="Number of species removed (r) ",
            ylabel="Number of species lost (n) ")

    ax1.axhline(0.5*S, color="black",
                linestyle="dotted")  # 50% of species lost
    ax1.set_xlim([0, dim+1])
    ax1.set_ylim([0, dim+1])

    # plt.legend()

    ax1.set(title="n= "+str(dim)+", Connectance= "+str(connectance))
    handles, labels = ax1.get_legend_handles_labels()
    return(handles, labels)
