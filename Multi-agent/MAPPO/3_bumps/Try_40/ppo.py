import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd

from torch.distributions import MultivariateNormal
from torch.optim import Adam
from actor_network import FeedForwardNN_act
from critic_network import FeedForwardNN_crit
from geometry_generator import generate_geometry_start

from os import path

class PPO:

  def __init__(self):

    # Call hyperparameters
    self._init_hyperparameters()
    
    reward_log_file = open("ref.log", "w") #write to file without line
    reward_log_file.write("t , center_1 , radius_1 , center_2, radius_2, center_3, radius_3, p_33, p , |((p_inl-p_0)/(p_33-p_0))-1| , T_33, T , (T_outl-T_0)/(T_33-T_0)\n")
    reward_log_file.close()
    
    # Generate domain with bump at min radius
    r_min_1, r_min_2, r_min_3 = generate_geometry_start(self.bump_center_1,self.bump_center_2,self.bump_center_3,self.bump_radius_min,self.bump_radius_min,self.bump_radius_min,self.bump_radius_min,self.bump_radius_max)

    # Run simulation with bump at max radius
    os.system("$MPIEXEC $FLAGS_MPI_BATCH /hpcwork/jara0203/RL/MAIA/Solver/src/maia properties_run.toml > maia_command.log")

    # Receive feedback from MAIA --> extract p_inl and T_outl
    min_p_inl_norm,min_T_outl_norm,min_p_inl,min_T_outl = self.response_from_maia()

    # Set new extrema for p_inl and T_outl
    self.p_0=min_p_inl
    self.T_0=min_T_outl
    
    #Move output files
    os.system("mv Output_SegNo_0_BC_1022.dat output_files/min_Output_SegNo_0_BC_1022.dat")
    os.system("mv Output_SegNo_1_BC_4130.dat output_files/min_Output_SegNo_1_BC_4130.dat")

    # Generate domain with bump at max radius
    r_max_1, r_max_2,r_max_3 = generate_geometry_start(self.bump_center_1,self.bump_center_2,self.bump_center_3,self.bump_radius_max,self.bump_radius_max,self.bump_radius_max,self.bump_radius_min,self.bump_radius_max)
     
    # Run simulation with bump at max radius
    os.system("$MPIEXEC $FLAGS_MPI_BATCH /hpcwork/jara0203/RL/MAIA/Solver/src/maia properties_run.toml > maia_command.log")

    # Receive feedback from MAIA --> extract p_inl and T_outl
    max_p_inl_norm,max_T_outl_norm,max_p_inl,max_T_outl = self.response_from_maia()
   
    # Set new extrema for p_inl and T_outl
    self.p_33=max_p_inl
    self.T_33=max_T_outl
       
    #Move output files
    os.system("mv Output_SegNo_0_BC_1022.dat output_files/max_Output_SegNo_0_BC_1022.dat")
    os.system("mv Output_SegNo_1_BC_4130.dat output_files/max_Output_SegNo_1_BC_4130.dat")

    
   
  def learn(self):

    t=2615
    rad_1 = np.arange(22.6, 29, 0.2, dtype=float)
    rad_2 = np.arange(24, 29, 0.2, dtype=float)
    rad_3 = np.arange(27.4, 29, 0.2, dtype=float)

    for i in rad_1:
        for j in rad_2:
            for k in rad_3:

                r_max_1, r_max_2,r_max_3 = generate_geometry_start(self.bump_center_1,self.bump_center_2,self.bump_center_3,i,j,k,self.bump_radius_min,self.bump_radius_max)  
    
                # Run simulation with bump at max radius
                os.system("$MPIEXEC $FLAGS_MPI_BATCH /hpcwork/jara0203/RL/MAIA/Solver/src/maia properties_run.toml > maia_command.log")

                # Receive feedback from MAIA --> extract p_inl and T_outl
                p_inl_norm,T_outl_norm,p_inl,T_outl = self.response_from_maia()

                reward_log_file = open("3_bumps_refined.log", "a") #write to file without line
                reward_log_file.write(str(t+1) + " , " + str(self.bump_center_1) + " , " + str(i) + " , " + str(self.bump_center_2) + " , " + str(j) + " , " + str(self.bump_center_3) + " , " + str(k) + " , " + str(self.p_33) + " , " + str(p_inl) + " , " + str(p_inl_norm) + " , " + str(self.T_33) + " , " + str(T_outl) + " , " + str(T_outl_norm) + "\n")
                reward_log_file.close()
            
                t=t+1
    
                #Remove output files
                os.system("rm Output_SegNo_0_BC_1022.dat")
                os.system("rm Output_SegNo_1_BC_4130.dat")
    
  #Define hyperparameters
  def _init_hyperparameters(self):

    #Bump parameters
    self.bump_center_1 = 75
    self.bump_center_2 = 125
    self.bump_center_3 = 175
    self.bump_radius_init = 10
    self.bump_radius_min = 0
    self.bump_radius_max = 33

    # Reward function
    self.p_0=0.3343346
    self.p_33=0.3376123   

    self.T_0=1.0390194
    self.T_33=1.0406535

  # Calculate p and T responses from output file
  def response_from_maia(self):
    
    # read output files
    p_stat_inl = pd.read_csv('Output_SegNo_0_BC_1022.dat', sep=" " , usecols = [3], engine='python')
    T_outl = pd.read_csv('Output_SegNo_1_BC_4130.dat', sep=" " , usecols = [5], engine='python')

    p_stat_inl = p_stat_inl[''+p_stat_inl.columns[0]+''][p_stat_inl.shape[0]-1]
    T_outl = T_outl[''+T_outl.columns[0]+''][T_outl.shape[0]-1]

    #Normalize p_inl and T_outl
    p_norm_inl=(p_stat_inl-self.p_0)/(self.p_33-self.p_0)
    T_norm_outl=(T_outl-self.T_0)/(self.T_33-self.T_0)

    return(p_norm_inl,T_norm_outl,p_stat_inl,T_outl)

   
model = PPO()
model.learn()

