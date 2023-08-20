import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from myTFMlibrary import *
from decompositionMod import ExtinctionExperimentNetworkx
from decompositionMod import ExtinctionExperimentNetworkxV2
from decompositionMod import Decomposition_Bipartite
import decomposition  #Carpenter's original codes library
import robustness   #Carpenter's original codes library
import timeit
import networkx as nx
import nestedness_metrics



def main():
    #VARIABLES DEF:
    path_figures="path for .csv files" 
    DIMMENTIONS=[10,30,60]
    CONNECTANCES=np.linspace(0.1,0.9,50)

    nbsimu = 100 # Number of different decompositions (simulations) to perform.
    loop=15
    Settings=["TablanConnectance"] 

    for S in DIMMENTIONS:
        nombretabla="/Connectance_Tabla_MOD_"+str(S)  #Name of the csv file
        print(nombretabla)
        for C in CONNECTANCES: 
            for r in range(loop):
                    # Generate a random binary matrix
                    #################################
                    mat= mat=symmetricBinaryMatrix(S,C)
        
 
                    # Decompositions (in-silico extinction experiments)
                    ###################################################
                    independent = True 
                    S, L, b, z, sseq, lseq = ExtinctionExperimentNetworkx(mat, nbsimu, independent)
                    
                    # R2 of the prediction of the L~S relationship 
                    ###############################################
                    breg, areg, r2reg, r2b = decomposition.R2L(sseq, lseq, b)
                    b_exp=breg
                    a_exp=areg
                    b_the=b

                    # Average number of extinctions occurring after the removal of one species
                    ###########################################################################
                    deltaS, r2lost, lost, removed = robustness.R2deltaS(sseq, S, b, z)
                    # deltaS = Predicted average number of extinctions after one removal (Eq. 4).
                    # r2lost = R2 of the predictions of the number of species lost 
                    #          based on the number of species removed using deltaS.    
                    # lost = Number of species lost along each decomposition (row).
                    # removed = Number of species removed along each decomposition (row).

                    # Robustness
                    ############
                    rob_obs, rob_var, rob_pred = robustness.robx(sseq, S, b, z, x=0.5)
                    rob_obs_reg, rob_var_reg, rob_pred_reg = robustness.robx(sseq, S, breg, z, x=0.5)
                    # rob_obs = Mean robustness over nbsimu network decompositions.   
                    # rob_var = Observed variance of robustness over nbsimu network decompositions.
                    # rob_pred = Predicted robustness (Equation 5).



                    ####################################################################################################
                    ####################################################################################################
                    #########################     Settings TablaRobustness        ######################################
                    ####################################################################################################
                    ####################################################################################################   

                    if "TablanConnectance" in Settings:
                        #Datasave Module
                        path_output= path_figures #Cambiar cuando se pase a linux.
                        try:
                            ReadTabla=pd.read_csv(path_output+nombretabla+".csv")
                            #print(nombretabla+".csv exist")
                        
                            df=pd.DataFrame({"Dimention":[S],
                                            "C":[C],
                                            "Predicted_Rob":[rob_pred],
                                            "Predicted_Rob_reg":[rob_pred_reg],
                                            "Observed_Rob":[rob_obs],
                                            "Observed_Rob_reg":[rob_obs_reg],
                                            "Predicted_b":[b_the],
                                            "Observed_b":[b_exp],
                                            "Observed_a":[a_exp],
                                            "DeltaS":[deltaS]})
                            
                            ReadTabla=pd.concat([ReadTabla,df])
                            ReadTabla.to_csv(path_output+nombretabla+".csv",index=False)
                            #print(nombretabla+".csv updated")


                        except Exception:
                            RobustnessDataframe=pd.DataFrame({"Dimention":[S],
                                            "C":[C],
                                            "Predicted_Rob":[rob_pred],
                                            "Predicted_Rob_reg":[rob_pred_reg],
                                            "Observed_Rob":[rob_obs],
                                            "Observed_Rob_reg":[rob_obs_reg],
                                            "Predicted_b":[b_the],
                                            "Observed_b":[b_exp],
                                            "Observed_a":[a_exp],
                                            "DeltaS":[deltaS]})
                            RobustnessDataframe.to_csv(path_output+nombretabla+".csv",index=False)
if __name__ == "__main__":
    main()