import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
import statistics

import torch.optim

from torch.distributions import MultivariateNormal
from actor_network_w_tan_1 import FeedForwardNN_act_1
from critic_network_1 import FeedForwardNN_crit_1
from actor_network_w_tan_2 import FeedForwardNN_act_2
from critic_network_2 import FeedForwardNN_crit_2
from geometry_generator import generate_geometry_start
from geometry_generator import generate_geometry_train

from os import path

class MAA2C:

  def __init__(self):

    # Extract environment information
    self.obs_dim = 4 #Changes as a result of actions [bump_radius]
    self.act_dim = 1 #number of actions=2  --> change two radii

    # Call hyperparameters
    self._init_hyperparameters()
    
    # Initialize actor and critic networks
    self.actor_1 = FeedForwardNN_act_1(self.obs_dim, self.act_dim)
    self.critic_1 = FeedForwardNN_crit_1(self.obs_dim*2, 1)
    self.actor_2 = FeedForwardNN_act_2(self.obs_dim, self.act_dim)
    self.critic_2 = FeedForwardNN_crit_2(self.obs_dim*2, 1)

    # Initialize optimizers for actor and critic
    self.actor_optim_1 = torch.optim.Adam(self.actor_1.parameters(), lr=self.lr, weight_decay=0.05)
    self.actor_optim_2 = torch.optim.Adam(self.actor_2.parameters(), lr=self.lr, weight_decay=0.05)
    #self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
    #self.actor_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.actor_optim, mode='max', factor=0.1, patience=8, min_lr = 0.00005)

    self.critic_optim_1 = torch.optim.Adam(self.critic_1.parameters(), lr=self.lr, weight_decay=0.05)
    self.critic_optim_2 = torch.optim.Adam(self.critic_2.parameters(), lr=self.lr, weight_decay=0.05)
    #self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
    #self.critic_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(self.critic_optim, mode='max', factor=0.1, patience=8, min_lr = 0.00005)

    # If restart, load actor and critic models
    if self.restart == 1:
   
       self.actor_1.load_state_dict(torch.load('./ppo_actor_1_000020.pth'))
       self.actor_optim_1.load_state_dict(torch.load('./ppo_actor_1_opt_000020.pth'))
       #self.actor_schedule.load_state_dict(torch.load('./ppo_actor_sch.pth'))
       self.critic_1.load_state_dict(torch.load('./ppo_critic_1_000020.pth'))
       self.critic_optim_1.load_state_dict(torch.load('./ppo_critic_1_opt_000020.pth'))
       #self.critic_schedule.load_state_dict(torch.load('./ppo_critic_sch.pth'))

       #self.actor_optim.param_groups[0]['lr'] = self.lr
       #self.critic_optim.param_groups[0]['lr'] = self.lr

       self.actor_2.load_state_dict(torch.load('./ppo_actor_2_000020.pth'))
       self.actor_optim_2.load_state_dict(torch.load('./ppo_actor_2_opt_000020.pth'))
       #self.actor_schedule.load_state_dict(torch.load('./ppo_actor_sch.pth'))
       self.critic_2.load_state_dict(torch.load('./ppo_critic_2_000020.pth'))
       self.critic_optim_2.load_state_dict(torch.load('./ppo_critic_2_opt_000020.pth'))
    # Else init log file if new start
    elif self.restart == 0:
    
       rl_log_file = open("rl.log", "w") #write to file without line
       rl_log_file.write("self.i_so_far, self.t_so_far, avg_ep_rews_1, avg_ep_rews_2, self.ep_rew_best, self.R_1_best, self.R_2_best, avg_actor_loss, avg_critic_loss, self.R_1_batch, self.R_1_sum, self.R_1_avg, self.R_2_batch, self.R_2_sum, self.R_2_avg, self.act_lr, self.cri_lr \n")
       rl_log_file.close()

       reward_log_file = open("reward_updated.log", "w") #write to file without line
       reward_log_file.write("self.i_so_far , t , center_1 , radius_1 , center_2, radius_2, p_33, p , |((p_inl-p_0)/(p_33-p_0))-1| , T_33, T , (T_outl-T_0)/(T_33-T_0) , rew_sum, rew \n")
       reward_log_file.close()

       reward_log_file = open("best_reward_updated.log", "w") #write to file without line
       reward_log_file.write("self.i_so_far, t, radius_1, radius_2, best_radius_1, best_radius_2, reward, rewards_1, rewards_2")
    # Initialize the covariance matrix used to query the actor for actions
    # The idea is to have the actor network output a "mean" action ona  forward pass,
    # then create a covariance matrix yielding a stddev along the diagonals.
    # Then we can use mean and stddev to generate a Multivariate Normal Distribution.
    # Then sample an action close to our mean.
    # Note that I chose 0.5 for stdev arbitrarily.
    self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
  
    # Create the covariance matrix
    self.cov_mat = torch.diag(self.cov_var)
    
    # Logger to monitor progress
    self.logger = {
           't_so_far':0,          # time steps so far
           'i_so_far':0,          # iterations so far
           'batch_lens': [],      # episodic lengths in batch
           'batch_rews_1': [],      # episodic returns in batch
           'batch_rews_2': [],      # episodic returns in batch
           'actor_losses_1': [],    #losses of actor network in current iteration
           'actor_losses_2': [],    #losses of actor network in current iteration
           'critic_losses_1': [],    #losses of critic network in current iteration
           'critic_losses_2': [],    #losses of critic network in current iteration
      }

    # Generate domain with bump at min radius
    r_min_1, r_min_2 = generate_geometry_start(self.bump_center_1,self.bump_center_2,self.bump_radius_min,self.bump_radius_min,self.bump_radius_min,self.bump_radius_max)

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
    # Generate geometry at max radius:
    r_max_1, r_max_2 = generate_geometry_start(self.bump_center_1,self.bump_center_2,self.bump_radius_max,self.bump_radius_max,self.bump_radius_min,self.bump_radius_max)

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

  def learn(self, total_timesteps):

    #print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
    #print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")
    #obs_init = []
    #Init avg qnd batch Radii
    self.R_1_sum=0
    self.R_2_sum=0
    self.R_1_avg=0
    self.R_2_avg=0
    self.R_1_batch=0
    self.R_2_batch=0
    self.ep_rew_best=0
    self.R_1_best=0
    self.R_2_best=0

    if self.restart == 0:

       self.t_so_far = 0 # Timesteps simulated so far
       self.i_so_far = 0 # Iterations ran so far
    
    elif self.restart == 1:

       time_steps_before_restart = np.loadtxt('rl.log', delimiter ="," , usecols =[1], skiprows=1)
       iterations_before_restart = np.loadtxt('rl.log', delimiter ="," , usecols =[0], skiprows=1)
       R_1_sum_before_restart = np.loadtxt('rl.log', delimiter ="," , usecols =[10], skiprows=1)
       R_2_sum_before_restart = np.loadtxt('rl.log', delimiter ="," , usecols =[13], skiprows=1)
       ep_rew_best_before_restart = np.loadtxt('rl.log', delimiter ="," , usecols =[4], skiprows=1)
       R_1_best_before_restart = np.loadtxt('rl.log', delimiter ="," , usecols =[5], skiprows=1)
       R_2_best_before_restart = np.loadtxt('rl.log', delimiter ="," , usecols =[6], skiprows=1)
       
       self.t_so_far = int(time_steps_before_restart[time_steps_before_restart.shape[0]-1]) # Timesteps simulated so far
       self.i_so_far = int(iterations_before_restart[iterations_before_restart.shape[0]-1]) # Iterations ran so far
       self.ep_rew_best = int(ep_rew_best_before_restart[iterations_before_restart.shape[0]-1])
       self.R_1_best = int(R_1_best_before_restart[iterations_before_restart.shape[0]-1])
       self.R_2_best = int(R_2_best_before_restart[iterations_before_restart.shape[0]-1])
       self.R_1_sum = int(R_1_sum_before_restart[iterations_before_restart.shape[0]-1])
       self.R_2_sum = int(R_2_sum_before_restart[iterations_before_restart.shape[0]-1])

       # update the learning rate
       #for i in range(self.i_so_far):
       #   self.actor_schedule.step() 
       #   self.critic_schedule.step() 

    #Initialize observation array  
    obs_1 = np.zeros((self.obs_dim))
    obs_2 = np.zeros((self.obs_dim))
    obs_init = np.zeros((self.obs_dim))
    #obs_01 = np.zeros((self.obs_dim))
    """
    self.reset_restart_property_file()
    # Generate domain with initial bump
    obs_init[0],obs_init[1] = generate_geometry_start(self.bump_center_1,self.bump_center_2,self.bump_radius_init,self.bump_radius_init,self.bump_radius_min,self.bump_radius_max)
 
    # Run simulation with initial bump
    os.system("$MPIEXEC $FLAGS_MPI_BATCH /hpcwork/jara0203/RL/MAIA/Solver/src/maia properties_run.toml > maia_command.log")
      
    obs_init[2], obs_init[3], p,T = self.response_from_maia()
     
    #Move output files
    os.system("rm Output_SegNo_0_BC_1022.dat")
    os.system("rm Output_SegNo_1_BC_4130.dat")
    print('obs_init', obs_init)
    """
    self.reset_restart_property_file()
    # Generate domain with initial bump
    obs_1[0],obs_1[1] = generate_geometry_start(self.bump_center_1,self.bump_center_2,self.bump_radius_init,self.bump_radius_init,self.bump_radius_min,self.bump_radius_max)
 
    # Run simulation with initial bump
    os.system("$MPIEXEC $FLAGS_MPI_BATCH /hpcwork/jara0203/RL/MAIA/Solver/src/maia properties_run.toml > maia_command.log")
      
    obs_1[2], obs_1[3], p,T = self.response_from_maia()
     
    #Move output files
    os.system("rm Output_SegNo_0_BC_1022.dat")
    os.system("rm Output_SegNo_1_BC_4130.dat")
    obs_init = np.copy(obs_1)
    #obs_init[0] = obs_1[0]
    #obs_init[1] = obs_1[1]
    #obs_init[2] = obs_1[2]
    #obs_init[3] = obs_1[3]
    rewards_1 = np.zeros((5)) 
    rewards_2 = np.zeros((5))
    best_radius_1 = np.zeros((5))
    best_radius_2 = np.zeros((5))
    # Overall loop
    while self.t_so_far < total_timesteps:    
    
      # Perform one batch with "time_steps_per_batch" time steps  
      batch_obs_1, batch_obs_2, batch_acts_1, batch_acts_2, batch_log_probs_1, batch_log_probs_2, batch_rtgs_1, batch_rtgs_2, batch_advs_1, batch_advs_2, batch_lens = self.rollout(obs_1,obs_2, obs_init, rewards_1, rewards_2, best_radius_1, best_radius_2)

      # Calculate how many timesteps we collected this batch   
      self.t_so_far += np.sum(batch_lens)

      # Increment the number of iterations
      self.i_so_far += 1

      # Logging timesteps so far and iterations so far
      self.logger['t_so_far'] = self.t_so_far
      self.logger['i_so_far'] = self.i_so_far

      # This is the loop where we update our network for some n epochs for AGENT 1
      for _ in range(self.n_updates_per_iteration):

        # Calculate V_phi and pi_theta(a_t | s_t)
        curr_log_probs_1, dist_entropies_1 = self.evaluate(batch_obs_1, batch_acts_1)

        # Calculate ratios
        #ratios_1 = torch.exp(curr_log_probs_1 - batch_log_probs_1)

        # Calculate surrogate losses
        #surr1_1 = ratios_1 * batch_advs_1
        #surr2_1 = torch.clamp(ratios_1, 1 - self.clip, 1 + self.clip) * batch_advs_1

        # actor and critic losses
        actor_loss_1 = (-torch.multiply(curr_log_probs_1, batch_advs_1)).mean()
        actor_loss_1 = actor_loss_1 - dist_entropies_1 * self.entr_loss_weight

        # Calculate gradients and perform backward propagation for actor network
        self.actor_optim_1.zero_grad()
        actor_loss_1.backward(retain_graph=True) #Only one backpropagation at a time
        nn.utils.clip_grad_norm_(self.actor_1.parameters(), 0.5) #Gradient clipping
        self.actor_optim_1.step()

        # Calculate gradients and perform backward propagation for critic network    
      
      # This is the loop where we update our network for some n epochs for AGENT 2
#      for _ in range(self.n_updates_per_iteration):

        # Calculate V_phi and pi_theta(a_t | s_t)
        curr_log_probs_2, dist_entropies_2 = self.evaluate(batch_obs_2, batch_acts_2)

        # Calculate ratios
        #ratios_2 = torch.exp(curr_log_probs_2 - batch_log_probs_2)

        # Calculate surrogate losses
        #surr1_2 = ratios_2 * batch_advs_2
        #surr2_2 = torch.clamp(ratios_2, 1 - self.clip, 1 + self.clip) * batch_advs_2

        # actor and critic losses
        actor_loss_2 = (-torch.multiply(curr_log_probs_2, batch_advs_2)).mean()
        actor_loss_2 = actor_loss_2 - dist_entropies_2 * self.entr_loss_weight

        # Calculate gradients and perform backward propagation for actor network
        self.actor_optim_2.zero_grad()
        actor_loss_2.backward(retain_graph=True) #Only one backpropagation at a time
        nn.utils.clip_grad_norm_(self.actor_2.parameters(), 0.5) #Gradient clipping
        self.actor_optim_2.step()
        
        #print('dimension of batch_obs_1 and batch_obs_2', batch_obs_1.size(), batch_obs_2.size())
        batch_obs = torch.cat((batch_obs_1, batch_obs_2), 1)
        V_1 = self.critic_1(batch_obs).squeeze()
        #print('dimension of batch_obs after calling value network', batch_obs_1.size(), batch_obs_2.size())
        critic_loss_1 = nn.MSELoss()(V_1, batch_rtgs_1)        
        V_2 = self.critic_2(batch_obs).squeeze()
        critic_loss_2 = nn.MSELoss()(V_2, batch_rtgs_2)        

        critic_loss = critic_loss_1 + critic_loss_2
        self.critic_optim_1.zero_grad()    
        self.critic_optim_2.zero_grad()    
        critic_loss.backward()    
#        critic_loss.backward()    
        nn.utils.clip_grad_norm_(self.critic_1.parameters(), 0.5) #Gradient clipping
        nn.utils.clip_grad_norm_(self.critic_2.parameters(), 0.5) #Gradient clipping
        self.critic_optim_1.step()
        # Calculate gradients and perform backward propagation for critic network    
        self.critic_optim_2.step()

        # Log actor loss
        self.logger['actor_losses_1'].append(actor_loss_1.detach())
        self.logger['critic_losses_1'].append(critic_loss_1.detach())
        self.logger['actor_losses_2'].append(actor_loss_2.detach())
        self.logger['critic_losses_2'].append(critic_loss_2.detach())

  
      # Get current learning rates
      self.actor_1_lr=self.actor_optim_1.param_groups[0]['lr']
      self.critic_1_lr=self.critic_optim_1.param_groups[0]['lr']
      self.actor_2_lr=self.actor_optim_2.param_groups[0]['lr']
      self.critic_2_lr=self.critic_optim_2.param_groups[0]['lr']

      # Print a summary of our training so far and update learning rate
      self._log_summary()

      # Save our model if it's time
      if self.i_so_far % self.save_freq == 0:
        
        torch.save(self.actor_1.state_dict(), './ppo_actor_1.pth')
        torch.save(self.actor_optim_1.state_dict(), './ppo_actor_1_opt.pth')
        #torch.save(self.actor_schedule.state_dict(), './ppo_actor_sch.pth')
        torch.save(self.critic_1.state_dict(), './ppo_critic_1.pth')
        torch.save(self.critic_optim_1.state_dict(), './ppo_critic_1_opt.pth')
        torch.save(self.actor_2.state_dict(), './ppo_actor_2.pth')
        torch.save(self.actor_optim_2.state_dict(), './ppo_actor_2_opt.pth')
        #torch.save(self.actor_schedule.state_dict(), './ppo_actor_sch.pth')
        torch.save(self.critic_2.state_dict(), './ppo_critic_2.pth')
        torch.save(self.critic_optim_2.state_dict(), './ppo_critic_2_opt.pth')
        #torch.save(self.critic_schedule.state_dict(), './ppo_critic_sch.pth')
      
      # Make backups if it's time
      if self.i_so_far % self.backup_freq == 0:
        
        torch.save(self.actor_1.state_dict(), './ppo_actor_1_'+'%06d'%self.i_so_far+'.pth')
        torch.save(self.actor_optim_1.state_dict(), './ppo_actor_1_opt_'+'%06d'%self.i_so_far+'.pth')
        #torch.save(self.actor_schedule.state_dict(), './ppo_actor_sch_'+'%06d'%self.i_so_far+'.pth')
        torch.save(self.critic_1.state_dict(), './ppo_critic_1_'+'%06d'%self.i_so_far+'.pth')
        torch.save(self.critic_optim_1.state_dict(), './ppo_critic_1_opt_'+'%06d'%self.i_so_far+'.pth')
        #torch.save(self.critic_schedule.state_dict(), './ppo_critic_sch_'+'%06d'%self.i_so_far+'.pth')
  
        torch.save(self.actor_2.state_dict(), './ppo_actor_2_'+'%06d'%self.i_so_far+'.pth')
        torch.save(self.actor_optim_2.state_dict(), './ppo_actor_2_opt_'+'%06d'%self.i_so_far+'.pth')
        #torch.save(self.actor_schedule.state_dict(), './ppo_actor_sch_'+'%06d'%self.i_so_far+'.pth')
        torch.save(self.critic_2.state_dict(), './ppo_critic_2_'+'%06d'%self.i_so_far+'.pth')
        torch.save(self.critic_optim_2.state_dict(), './ppo_critic_2_opt_'+'%06d'%self.i_so_far+'.pth')

  # Collect data for a batch
  def rollout(self,obs_1,obs_2, obs_init, rewards_1, rewards_2, best_radius_1, best_radius_2):

    # Initialize lists
    batch_obs_1 = []             # batch observations [#ts per batch, dim of obs]
    batch_obs_2 = []             # batch observations [#ts per batch, dim of obs]
    batch_acts_1 = []            # batch actions [#ts per batch, dim of action]
    batch_acts_2 = []            # batch actions [#ts per batch, dim of action]
    batch_log_probs_1 = []       # log probs of each action [#ts per batch]
    batch_log_probs_2 = []       # log probs of each action [#ts per batch]
    batch_rews_1 = []            # batch rewards [#episodes, #ts per episode]
    batch_rews_2 = []            # batch rewards [#episodes, #ts per episode]
    batch_vals_1 = []            # batch values [#episodes, #ts per episode]
    batch_vals_2 = []            # batch values [#episodes, #ts per episode]
    batch_advs_1 = []            # batch advantageso [#ts per batch]
    batch_advs_2 = []            # batch advantageso [#ts per batch]
    batch_lens = []            # episodic lengths in batch [#episodes]

    # Number of timesteps run so far this batch
    t = 0
    #obs_1 = obs_init
    #if ts == 1: # infinite loop
    # Keep simulating until we've run more than or equal to specified timesteps per batch
    while t < self.timesteps_per_batch:
  
      # Rewards this episode
      ep_rews_1 = []
      ep_rews_2 = []
 
      # Values this episode
      ep_vals_1 = []
      ep_vals_2 = []
      ep_rew_best_1 = []
      ep_rew_best_2 = []
 
      ts = 1
      # Update properties_restart.toml

      #Init flag
      done = False

      # Run an episode for a maximum of max_time_steps_per_episode timesteps
      for ep_t in range(self.max_timesteps_per_episode):

        # Increment timesteps ran this batch so far
        t += 1

        # List for target radius
        R_1_list = []
        R_2_list = []
    
        # AGENT 1
        # Collect observation
        #print('The dimension of obs_1 is ',len(obs_1) ) 
        batch_obs_1.append(obs_1)
        print('batch_obs_1', batch_obs_1)
        last_obs_1 = batch_obs_1[-1]
        print('last_obs_1', last_obs_1)
        action_1, log_prob_1 = self.get_action(obs_1,1) # action --> get output from actor network, log_prob --> get the prob distribution of actions
        
        # Collect value
        action_tensor_1 = torch.tensor(action_1, dtype=torch.float)
        radius_change_1=action_1*3
        radius_change_2=0
        
        # Generate new domain
        obs_2[0],obs_2[1],death_max1,death_min1, death_max2, death_min2 = generate_geometry_train(self.i_so_far,t,self.bump_center_1,self.bump_center_2,batch_obs_1[-1][0],batch_obs_1[-1][1],radius_change_1,radius_change_2,self.bump_radius_min,self.bump_radius_max)

        #else:
        #Identify by what extend radius shall be changed --> output from actornetwor is [-1:1] with tanh, multiply by factor 3 to bound radius_change [-3;3]
        #A bound is necessary here, to avoid large jumps for more complex geometries later, that might cause computational problems

        #Denormalize radius
        
        # Run simulation
        os.system("$MPIEXEC $FLAGS_MPI_BATCH /hpcwork/jara0203/RL/MAIA/Solver/src/maia properties_restart.toml > maia_command.log")
        
        # Update properties_restart.toml
        self.update_restart_property_file(ts)
        ts += 1

        #ive response (delta p, delta t)  
        obs_2[2], obs_2[3], p, T = self.response_from_maia()
        obs_11 = np.concatenate((obs_1, obs_2), axis=None)
        V_1 = self.critic_1(obs_11).squeeze()
        #Move output files 
        os.system("rm Output_SegNo_0_BC_1022.dat")
        os.system("rm Output_SegNo_1_BC_4130.dat")
        
        #Reward: maximize |((p_inl-p_0)/(p_33-p_0))-1| + (T_outl-T_0)/(T_33-T_0) 
        # With a max value of 1.35
        rew=0
        rew_p=np.abs(obs_2[2]-1) #pressure
        rew_T=obs_2[3] #Temperature       
        rew_max=rew_p+rew_T
        #rew=1.35-rew_max #Subtract from maximum
        #Scale reward
        rew=rew_max*rew_max*rew_max*rew_max*rew_max
        
        #Normalize reward
        #rew=rew/1.35
        rew=rew/(1.35*1.35*1.35*1.35*1.35)
        
        #Gap between rewards
        #rew=rew*10

        radius_denorm_1=obs_2[0]*(self.bump_radius_max-self.bump_radius_min)+self.bump_radius_min
        radius_denorm_2=obs_2[1]*(self.bump_radius_max-self.bump_radius_min)+self.bump_radius_min
        # Collect statistics for best ep rew and the corresponding radii
        if rew > self.ep_rew_best:
           self.ep_rew_best=rew
           self.R_1_best = radius_denorm_1
           self.R_2_best = radius_denorm_2

           
        if death_max1 == True:
           rew = 0
	   #rew = rew/(1.35*1.35*1.35*1.35*1.35)
           #rew = self.bump_radius_max - obs_2[0]
           #obs_2[2] = 0
           #obs_2[3] = 0
           
           #obs_1 = batch_obs_1[len(batch_obs_1)-2]
           #obs_2 = batch_obs_1[0]
           #obs_2[0] = self.bump_radius_init
           #obs_2[1] = self.bump_radius_init
           
           #obs_2[2] = obs_init[2]#batch_obs_1[0][2]
           #obs_2[3] = obs_init[3]#batch_obs_1[0][3]
           #print(batch_obs_1[0])
           #action_1 = batch_acts_1[0]
           #batch_acts_1.append(action_1)
           #ep_rews_1.append(rew)
           #print('episode rewards inside the death due to radius max 1', ep_rews_1)
           
           #obs_11 = np.concatenate((obs_1, obs_2), axis=None)
           #V_1 = self.critic_1(obs_11).squeeze()
           #ep_vals_1.append(V_1)
           #print('episode values inside the death due to radius max 1', ep_vals_1)
           #batch_log_probs_1.append(log_prob_1)
           
           obs_2[0] = obs_init[0]
           obs_2[1] = obs_init[1]
           obs_2[2] = obs_init[2]
           obs_2[3] = obs_init[3]
           obs_1[0] = obs_init[0]
           obs_1[1] = obs_init[1]
           obs_1[2] = obs_init[2]
           obs_1[3] = obs_init[3]
           #log_prob_1 = batch_log_probs_1[0] 
           #continue
           radius_denorm_1 = self.bump_radius_init
           radius_denorm_2 = self.bump_radius_init
           #p = 0
           #T = 0
           #rew_p = 0
           #rew_T = 0
           #rew_max = 0
           #ts += 1
           #Write to reward.log file
           #reward_log_file = open("reward_death_2.log", "a") #write to file without line
           #reward_log_file.write(str(self.i_so_far) + " , " + str(t) + " , " + str(self.bump_center_1) + " , " + str(radius_denorm_1) + " , " + str(self.bump_center_2) + " , " + str(radius_denorm_2) + " , " + str(self.p_33) + " , " + str(p) + " , " +str(rew_p) + " , " + str(self.T_33) + " , " + str(T) + " , " + str(rew_T) + " , " + str(rew_max) + " , " + str(rew) + "\n")
           #reward_log_file.close()
           done = True
           

        elif death_min1 == True:
           rew = 0
           #rew = rew/(1.35*1.35*1.35*1.35*1.35)
           #rew = obs_2[0] - self.bump_radius_min
           #obs_1 = batch_obs_1[len(batch_obs_1)-2]
           #obs_2 = batch_obs_1[0]
           #obs_2[0] = self.bump_radius_init
           #obs_2[2] = obs_init[2]
           #obs_2[3] = obs_init[3]
           #obs_2[1] = self.bump_radius_init
           #obs_2[2] = batch_obs_1[0][2]
           #obs_2[3] = batch_obs_1[0][3]
           #action_1 = batch_acts_1[0]
           #batch_acts_1.append(action_1)
           #ep_rews_1.append(rew)
           #print('episode rewards inside the death due to radius min 1', ep_rews_1)
           
           #obs_11 = np.concatenate((obs_1, obs_2), axis=None)
           #V_1 = self.critic_1(obs_11).squeeze()
           #ep_vals_1.append(V_1)
           #print('episode values inside the death due to radius min 1', ep_vals_1)
           #log_prob_1 = batch_log_probs_1[0] 
           #batch_log_probs_1.append(log_prob_1)
           obs_2[0] = obs_init[0]
           obs_2[1] = obs_init[1]
           obs_2[2] = obs_init[2]
           obs_2[3] = obs_init[3]
           obs_1[0] = obs_init[0]
           obs_1[1] = obs_init[1]
           obs_1[2] = obs_init[2]
           obs_1[3] = obs_init[3]
           radius_denorm_1 = self.bump_radius_init
           radius_denorm_2 = self.bump_radius_init
           #p = 0
           #T = 0
           #rew_p = 0
           #rew_T = 0
           #rew_max = 0
           #log_prob_1 = batch_log_probs_1[0] 
           #continue
           #ts += 1
           #Write to reward.log file
           #reward_log_file = open("reward_death_2.log", "a") #write to file without line
           #reward_log_file.write(str(self.i_so_far) + " , " + str(t) + " , " + str(self.bump_center_1) + " , " + str(radius_denorm_1) + " , " + str(self.bump_center_2) + " , " + str(radius_denorm_2) + " , " + str(self.p_33) + " , " + str(p) + " , " +str(rew_p) + " , " + str(self.T_33) + " , " + str(T) + " , " + str(rew_T) + " , " + str(rew_max) + " , " + str(rew) + "\n")
           #reward_log_file.close()
           done = True
        
        if ep_t < 5:
           rewards_1[ep_t] = rew
           print(rewards_1)
           best_radius_1[ep_t] = radius_denorm_1
           #is_full_1 = rewards_1.size == 5
           #if is_full_1:
        if ep_t >= 5:
           idx_1 = np.argmin(rewards_1)
           #if rew <= min_1:
           min_1 = rewards_1[idx_1]
           if rew > min_1: 
              rewards_1[idx_1] = rew
              best_radius_1[idx_1] = radius_denorm_1
              #idx_1 = rewards_1.index(rew)
              #rewards_1.remove(rew)
              #batch_obs_1.pop(idx_1)
           else:
              rewards_1[idx_1] = min_1
        best_reward_log_file = open("best_reward_updated.log", "a") #write to file without line
        best_reward_log_file.write(str(self.i_so_far) + " , " + str(t) + " , " + str(radius_denorm_1) + " , " + str(radius_denorm_2) + " , " + str(best_radius_1) + " , " + str(best_radius_2) + " , " + str(rew) + " , " + str(rewards_1) + " , " + str(rewards_2) + "\n")
        best_reward_log_file.close()
        #Write to reward.log file
        reward_log_file = open("reward_updated.log", "a") #write to file without line
        reward_log_file.write(str(self.i_so_far) + " , " + str(t) + " , " + str(self.bump_center_1) + " , " + str(radius_denorm_1) + " , " + str(self.bump_center_2) + " , " + str(radius_denorm_2) + " , " + str(self.p_33) + " , " + str(p) + " , " +str(rew_p) + " , " + str(self.T_33) + " , " + str(T) + " , " + str(rew_T) + " , " + str(rew_max) + " , " + str(rew) + "\n")
        reward_log_file.close()

        #flip sign to distinguish R_1 and R_2 for multi agent training
        #Define when task is done --> Later version
        #obs_2=-obs_2

        # Collect reward, value, action, and log prob
        ep_rews_1.append(rew)
        print('episode reward outside the death agent 1', ep_rews_1)
        ep_vals_1.append(V_1)
        print('episode values outside the death agent 1', ep_vals_1)
        #print('The dimension of action_1 is ', len(action_1))
        batch_acts_1.append(action_1)
        batch_log_probs_1.append(log_prob_1)
        #if done:
        #   t=t+self.max_timesteps_per_episode-(ts-1)
        #   break
        
        #AGENT 2
        # Collect observation 
        batch_obs_2.append(obs_2)
        last_obs_2 = batch_obs_2[-1]
        #Irint(last_obs_2)
        action_2, log_prob_2 = self.get_action(obs_2,2) # action --> get output from actor network, log_prob --> get the prob distribution of actions
        
        # Collect value
        action_tensor_2 = torch.tensor(action_2, dtype=torch.float)
        radius_change_1=0
        radius_change_2=action_2*3

        # Generate new domain
        obs_1[0],obs_1[1],death_max1, death_min1, death_max2, death_min2 = generate_geometry_train(self.i_so_far,t,self.bump_center_1,self.bump_center_2,batch_obs_2[-1][0],batch_obs_2[-1][1],radius_change_1,radius_change_2,self.bump_radius_min,self.bump_radius_max)
        #else:
        #Identify by what extend radius shall be changed --> output from actornetwor is [-1:1] with tanh, multiply by factor 3 to bound radius_change [-3;3]
        #A bound is necessary here, to avoid large jumps for more complex geometries later, that might cause computational problems

        #Denormalize radius
        
        # Run simulation
        os.system("$MPIEXEC $FLAGS_MPI_BATCH /hpcwork/jara0203/RL/MAIA/Solver/src/maia properties_restart.toml > maia_command.log")
        
        # Update properties_restart.toml
        self.update_restart_property_file(ts)
        ts += 1

        #Receive response (delta p, delta t)  
        obs_1[2], obs_1[3], p, T = self.response_from_maia()
        
        obs_22 = np.concatenate((obs_2, obs_1), axis=None)
        V_2 = self.critic_2(obs_22).squeeze()
        #Move output files 
        os.system("rm Output_SegNo_0_BC_1022.dat")
        os.system("rm Output_SegNo_1_BC_4130.dat")
        
        #Reward: maximize |((p_inl-p_0)/(p_33-p_0))-1| + (T_outl-T_0)/(T_33-T_0) 
        # With a max value of 1.35
        rew=0
        rew_p=np.abs(obs_1[2]-1) #pressure
        rew_T=obs_1[3] #Temperature       
        rew_max=(rew_p+rew_T)
        #rew=1.35-rew_max #Subtract from maximum

        #Scale reward
        rew=rew_max*rew_max*rew_max*rew_max*rew_max
        #rew = 2*rew
        
        #Normalize reward
        #rew=rew_max/1.35
        rew=rew/(1.35*1.35*1.35*1.35*1.35)
        rew = 2*rew
        #Gap between rewards
        #rew=rew*10

        radius_denorm_3=obs_1[0]*(self.bump_radius_max-self.bump_radius_min)+self.bump_radius_min
        radius_denorm_4=obs_1[1]*(self.bump_radius_max-self.bump_radius_min)+self.bump_radius_min
        # Collect statistics for best ep rew and the corresponding radii
        if rew > self.ep_rew_best:
           self.ep_rew_best=rew
           self.R_1_best = radius_denorm_3
           self.R_2_best = radius_denorm_4

        if death_max2 == True:
           #rew = 0
           rew = self.bump_radius_max - obs_1[1]
           #obs_1[0] = self.bump_radius_init
           #obs_1[1] = self.bump_radius_init
           #obs_1[2] = obs_init[2]
           #obs_1[3] = obs_init[3]
           #obs_1[1] = self.bump_radius_init
           #obs_1[2] = batch_obs_1[0][2]
           #obs_1[3] = batch_obs_1[0][3]
           #obs_1 = batch_obs_2[0]
           #action_2 = batch_acts_2[0]
           #done = True
           #batch_acts_2.append(action_2)
           #ep_rews_2.append(rew)
           #print('episode rewards inside the death due to radius max 2', ep_rews_2)
           #obs_22 = np.concatenate((obs_2, obs_1), axis=None)
           #V_2 = self.critic_2(obs_22).squeeze()
           #ep_vals_2.append(V_2)
           #print('episode values inside the death due to radius max 2', ep_vals_2)
           #log_prob_2 = batch_log_probs_2[0]
           #batch_log_probs_2.append(log_prob_2)
           obs_2[0] = obs_init[0]
           obs_2[1] = obs_init[1]
           obs_2[2] = obs_init[2]
           obs_2[3] = obs_init[3]
           obs_1[0] = obs_init[0]
           obs_1[1] = obs_init[1]
           obs_1[2] = obs_init[2]
           obs_1[3] = obs_init[3]
           #log_prob_1 = batch_log_probs_1[0] 
           #continue
           radius_denorm_3 = self.bump_radius_init
           radius_denorm_4 = self.bump_radius_init
           #p = 0
           #T = 0
           #rew_p = 0
           #rew_T = 0
           #rew_max = 0
           #log_prob_1 = batch_log_probs_1[0] 
           #continue
           #ts += 1
           #Write to reward.log file
           #reward_log_file = open("reward_death_2.log", "a") #write to file without line
           #reward_log_file.write(str(self.i_so_far) + " , " + str(t) + " , " + str(self.bump_center_1) + " , " + str(radius_denorm_1) + " , " + str(self.bump_center_2) + " , " + str(radius_denorm_2) + " , " + str(self.p_33) + " , " + str(p) + " , " +str(rew_p) + " , " + str(self.T_33) + " , " + str(T) + " , " + str(rew_T) + " , " + str(rew_max) + " , " + str(rew) + "\n")
           #reward_log_file.close()
           done = True

        elif death_min2 == True:        
           #rew = 0
           rew = obs_1[1] - self.bump_radius_min
           #obs_1[0] = self.bump_radius_init
           #obs_1[1] = self.bump_radius_init
           #obs_1[2] = obs_init[2]
           #obs_1[3] = obs_init[3]
           #obs_1[1] = self.bump_radius_init
           #obs_1[2] = batch_obs_1[0][2]
           #obs_1[3] = batch_obs_1[0][3]
           #obs_1 = batch_obs_2[0]
           #action_2 = batch_acts_2[0]
           #log_prob_2 = batch_log_probs_2[0]
           #done = True
           #batch_acts_2.append(action_2)
           #ep_rews_2.append(rew)
           #print('episode rewards inside the death due to radius min 2', ep_rews_2)
           #obs_22 = np.concatenate((obs_2, obs_1), axis=None)
           #V_2 = self.critic_2(obs_22).squeeze()
           #ep_vals_2.append(V_2)
           #print('episode values inside the death due to radius min 2', ep_vals_2)
           #batch_log_probs_2.append(log_prob_2)
           obs_2[0] = obs_init[0]
           obs_2[1] = obs_init[1]
           obs_2[2] = obs_init[2]
           obs_2[3] = obs_init[3]
           obs_1[0] = obs_init[0]
           obs_1[1] = obs_init[1]
           obs_1[2] = obs_init[2]
           obs_1[3] = obs_init[3]
           #log_prob_1 = batch_log_probs_1[0] 
           #continue
           radius_denorm_3 = self.bump_radius_init
           radius_denorm_4 = self.bump_radius_init
           #p = 0
           #T = 0
           #rew_p = 0
           #rew_T = 0
           #rew_max = 0
           ##log_prob_1 = batch_log_probs_1[0] 
           #continue
           #ts += 1
           #Write to reward.log file
           #reward_log_file = open("reward_death_2.log", "a") #write to file without line
           #reward_log_file.write(str(self.i_so_far) + " , " + str(t) + " , " + str(self.bump_center_1) + " , " + str(radius_denorm_1) + " , " + str(self.bump_center_2) + " , " + str(radius_denorm_2) + " , " + str(self.p_33) + " , " + str(p) + " , " +str(rew_p) + " , " + str(self.T_33) + " , " + str(T) + " , " + str(rew_T) + " , " + str(rew_max) + " , " + str(rew) + "\n")
           #reward_log_file.close()
           done = True
        if ep_t < 5:
           rewards_2[ep_t] = rew
           print(rewards_1)
           best_radius_2[ep_t] = radius_denorm_4
           #is_full_1 = rewards_1.size == 5
           #if is_full_1:
        if ep_t >= 5:
           idx_2 = np.argmin(rewards_2)
           #if rew <= min_2:
           min_2 = rewards_2[idx_2]
           if rew > min_2:
              rewards_2[idx_2] = rew
              best_radius_2[idx_2] = radius_denorm_4
              #idx_1 = rewards_1.index(rew)
              #rewards_1.remove(rew)
              #batch_obs_1.pop(idx_1)
           else:
              rewards_2[idx_2] = min_2
        best_reward_log_file = open("best_reward_updated.log", "a") #write to file without line
        best_reward_log_file.write(str(self.i_so_far) + " , " + str(t) + " , " + str(radius_denorm_1) + " , " + str(radius_denorm_2) + " , " + str(best_radius_1) + " , " + str(best_radius_2) + " , " + str(rew) + " , " + str(rewards_1) + " , " + str(rewards_2) + "\n")
        best_reward_log_file.close()
        
        #Write to reward.log file
        reward_log_file = open("reward_updated.log", "a") #write to file without line
        reward_log_file.write(str(self.i_so_far) + " , " + str(t) + " , " + str(self.bump_center_1) + " , " + str(radius_denorm_1) + " , " + str(self.bump_center_2) + " , " + str(radius_denorm_2) + " , " + str(self.p_33) + " , " + str(p) + " , " +str(rew_p) + " , " + str(self.T_33) + " , " + str(T) + " , " + str(rew_T) + " , " + str(rew_max) + " , " + str(rew) + "\n")
        reward_log_file.close()

        # Define when task is done --> Later version
        # Collect reward, value, action, and log prob
        ep_rews_2.append(rew)
        print('episode rewards outside the death agent 2', ep_rews_2)
        ep_vals_2.append(V_2)
        print('episode values outside the death agent 2', ep_vals_2)
        batch_acts_2.append(action_2)
        batch_log_probs_2.append(log_prob_2)
        # Get target radius of episode
        R_1_list.append(radius_denorm_1)
        R_2_list.append(radius_denorm_2)
        
        #if done:
        #   t = t + self.max_timesteps_per_episode - (ts-1)
        #   break

      # Collect episodic length, rewards and values
      batch_lens.append(ep_t + 1) # plus 1 because timestep starts at 0
      batch_rews_1.append(ep_rews_1)
      print('batch rewards 1', batch_rews_1)
      batch_rews_2.append(ep_rews_2)
      print('batch rewards 2', batch_rews_2)
      batch_vals_1.append(ep_vals_1)
      print('batch values 1', batch_vals_1)
      batch_vals_2.append(ep_vals_2)
      print('batch values 2', batch_vals_2)

    #Average target radii of all episodes and calculate statistics
    self.R_1_batch=np.mean(np.array(R_1_list))
    self.R_2_batch=np.mean(np.array(R_2_list))
     
    self.R_1_sum=self.R_1_sum+self.R_1_batch
    self.R_2_sum=self.R_2_sum+self.R_2_batch

    self.R_1_avg=self.R_1_sum/(self.i_so_far+1)
    self.R_2_avg=self.R_2_sum/(self.i_so_far+1)

    # Reshape data as tensors in the shape specified before returning
    batch_obs_1 = torch.tensor(batch_obs_1, dtype=torch.float)
    batch_obs_2 = torch.tensor(batch_obs_2, dtype=torch.float)
    batch_acts_1 = torch.tensor(batch_acts_1, dtype=torch.float)
    batch_acts_2 = torch.tensor(batch_acts_2, dtype=torch.float)
    batch_log_probs_1 = torch.tensor(batch_log_probs_1, dtype=torch.float)
    batch_log_probs_2 = torch.tensor(batch_log_probs_2, dtype=torch.float)
    
    # Compute rewards to go
    batch_rtgs_1, batch_advs_1 = self.compute_advantage(batch_rews_1,batch_vals_1)
    batch_rtgs_2, batch_advs_2 = self.compute_advantage(batch_rews_2,batch_vals_2)
    
    # Log the episodic returns and episodic lengths in this batch.
    self.logger['batch_rews_1'] = batch_rews_1
    self.logger['batch_rews_2'] = batch_rews_2
    self.logger['batch_lens'] = batch_lens

    # Return the batch data
    return batch_obs_1, batch_obs_2, batch_acts_1, batch_acts_2, batch_log_probs_1, batch_log_probs_2, batch_rtgs_1, batch_rtgs_2, batch_advs_1, batch_advs_2, batch_lens

  def compute_advantage(self, batch_rews, batch_vals):
  
    # If no GAE, simply calculate rewards to go
    if self.gae==False:

       # The rewards-to-go (rtg) per episode per batch to return.
       # The shape will be (num timesteps per episode)
       batch_rtgs = []

       # Iterate through each episode backwards
       for ep_rews in reversed(batch_rews):
    
         discounted_reward = 0 # The discounted reward so far

         for rew in reversed(ep_rews):
    
           discounted_reward = rew + discounted_reward * self.gamma
           batch_rtgs.insert(0, discounted_reward)

    # If GAE, ...
    elif self.gae==True:

       # The rewards-to-go (rtg) per episode per batch to return.
       # The shape will be (num timesteps per episode)
       batch_rtgs = []
       
       # Iterate through each episode backwards
       for i in reversed(range(len(batch_rews))):
    
         last_gae = 0 # The last gae
         last_val = batch_rews[i][-1] # The last value

         for j in reversed(range(len(batch_rews[i]))):
 
            delta = batch_rews[i][j] + self.gamma * last_val - batch_vals[i][j]
            gae = delta + self.gamma * self.lam * last_gae
            batch_rtgs.insert(0, gae+batch_vals[i][j])

            last_gae = gae
            last_val = batch_vals[i][j]        
          
    # Convert the rewards-to-go to a tensor
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
    
    # Normalize rtgs
    #batch_rtgs = (batch_rtgs - batch_rtgs.mean()) / (batch_rtgs.std() + 1e-10)
    
    # List for batch values in correct order    
    batch_vals_list = []    
    
    # Iterate through each episode and store as tensor
    for ep_vals in(batch_vals):
  
       for val in(ep_vals):

          batch_vals_list.append(val)      

    batch_vals = torch.tensor(batch_vals_list, dtype=torch.float)

    # Calculate advantage
    A = batch_rtgs - batch_vals.detach() #detach means that there will be no gradient update

    # Normalize advantages
    A = (A - A.mean()) / (A.std() + 1e-10)
    
    return batch_rtgs, A
   
  def get_action(self, obs, agent):

    # Query the actor network for a mean action.
    
    # if agent==1:
        
    mean = self.actor_1(obs)
    # Create our Multivariate Normal Distribution
    dist = MultivariateNormal(mean, self.cov_mat)
    # Sample an action from the distribution and get its log prob
    action = dist.sample()
    
    #else:
       
    #   mean = self.actor_2(obs)
    #   # Create our Multivariate Normal Distribution
    #   dist = MultivariateNormal(mean, self.cov_mat)
    #   # Sample an action from the distribution and get its log prob
    #   action = dist.sample()


    log_prob = dist.log_prob(action)
  
    # Return the sampled action and the log prob of that action
    # Note that I'm calling detach() since the action and log_prob  
    # are tensors with computation graphs, so I want to get rid
    # of the graph and just convert the action to numpy array.
    # log prob as tensor is fine. Our computation graph will
    # start later down the line.
    return action.detach().numpy(), log_prob.detach() 

  def evaluate(self, batch_obs, batch_acts):
    # Query critic network for a value V for each obs in batch_obs.
    #V_1 = self.critic_1(batch_obs_1, batch_obs_2).squeeze() #For example, calling squeeze on [[1], [2], [3]] will return [1, 2, 3]
    #V_2 = self.critic_2(batch_obs_2, batch_obs_1).squeeze() #For example, calling squeeze on [[1], [2], [3]] will return [1, 2, 3]

    # Calculate the log probabilities of batch actions using most 
    # recent actor network.
    # This segment of code is similar to that in get_action()
    mean = self.actor_1(batch_obs)
    dist = MultivariateNormal(mean, self.cov_mat)
    log_probs = dist.log_prob(batch_acts)
    
    #Calculate entropy
    dist_entropy = dist.entropy().mean()

    # Return predicted values V, log probs and entropies
    return log_probs, dist_entropy

  #Define hyperparameters
  def _init_hyperparameters(self):

    # We control the learning process with "total_timesteps" as input to the main function "learn()" 
    # Learning continues until "total_timesteps" is reached
    
    # Before training actor and critics we need to give different geometries to try out different things
    # We call such a trial and error phase in general a "batch"
    # We define how many time steps are trained in each batch
    self.timesteps_per_batch = 30  #4800
    
    # For each newly generated geometry in a batch the agent has a certain number of time steps to try things out (change the shape)
    # We call such a trial and error phase with a new geometry an "episode"
    # We define for how many time steps the agent can try out things in each episode
    self.max_timesteps_per_episode = 10 #1600

    self.gamma = 0.98 # As recommended by the paper
    self.n_updates_per_iteration = 5 #usually between 5 and 20
    #self.clip = 0.2 # As recommended by the paper
    self.lr = 0.1
    self.save_freq = 1 # How often we save in number of iterations
    self.backup_freq = 10 # How often we backup in number of iterations
    self.restart = 1 
    #self.steps=80

    self.entr_loss_weight=0.01
    #self.entr_discount_rate=0.99

    self.gae=True
    self.lam=0.95 #GAE parameter

    #Bump parameters
    self.bump_center_1 = 75
    self.bump_center_2 = 125
    self.bump_radius_init = 10
    self.bump_radius_min = 0
    self.bump_radius_max = 33

    # We need to define the MAIA time step in here
    self.maia_time_steps = 150000
    self.maia_init_time_steps = 300000

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

  # Reset properties_restart.toml after new geometry
  def reset_restart_property_file(self):

    a_file = open("properties_restart.toml", "r") #get list of lines
    lines = a_file.readlines()
    a_file.close()

    # Update lines in property file here, if file changes
    lines[42]=""" restartFileName = "restart_""" + str(self.maia_init_time_steps) + """.Netcdf" \n""" #update restartFileName
    lines[171]=" restartTimeStep = " + str(self.maia_init_time_steps) + " \n" #update restartTimeStep
    
    new_file = open("properties_restart.toml", "w+") #write to file without line
    for line in lines:
       new_file.write(line)

    new_file.close()
  
  # Update properties_restart.toml
  def update_restart_property_file(self,ts):

    a_file = open("properties_restart.toml", "r") #get list of lines
    lines = a_file.readlines()
    a_file.close()

    # Update lines in property file here, if file changes
    lines[42]=""" restartFileName = "restart_""" + str(self.maia_init_time_steps+ts*self.maia_time_steps) + """.Netcdf" \n""" #update restartFileName
    lines[171]=" restartTimeStep = " + str(self.maia_init_time_steps+ts*self.maia_time_steps) + " \n" #update restartTimeStep
    
    new_file = open("properties_restart.toml", "w+") #write to file without line
    for line in lines:
       new_file.write(line)

    new_file.close()

  #Logger
  def _log_summary(self):

    # Calculate logging values. I use a few python shortcuts to calculate each value
    # without explaining since it's not too important to PPO; feel free to look it over,
    # and if you have any questions you can email me (look at bottom of README)
    self.t_so_far = self.logger['t_so_far']
    self.i_so_far = self.logger['i_so_far']
    avg_ep_lens = np.mean(self.logger['batch_lens'])
    avg_ep_rews_1 = np.mean([np.sum(ep_rews_1) for ep_rews_1 in self.logger['batch_rews_1']])
    avg_ep_rews_2 = np.mean([np.sum(ep_rews_2) for ep_rews_2 in self.logger['batch_rews_2']])
    avg_actor_loss_1 = np.mean([losses.float().mean() for losses in self.logger['actor_losses_1']])
    avg_actor_loss_2 = np.mean([losses.float().mean() for losses in self.logger['actor_losses_2']])
    avg_actor_loss = 0.5*(avg_actor_loss_1+avg_actor_loss_2)
    avg_critic_loss_1 = np.mean([losses.float().mean() for losses in self.logger['critic_losses_1']])
    avg_critic_loss_2 = np.mean([losses.float().mean() for losses in self.logger['critic_losses_2']])
    avg_critic_loss = 0.5*(avg_critic_loss_1+avg_critic_loss_2)

    # Round decimal places for more aesthetic logging messages
    avg_ep_lens = str(round(avg_ep_lens, 2))
    avg_ep_rews_1 = str(round(avg_ep_rews_1, 2))
    avg_ep_rews_2 = str(round(avg_ep_rews_2, 2))
    avg_actor_loss = str(round(avg_actor_loss, 5))
    avg_critic_loss = str(round(avg_critic_loss, 5))

    #Write to log file
    rl_log_file = open("rl.log", "a") #write to file without line
    rl_log_file.write(str(self.i_so_far) + " , " + str(self.t_so_far) + " , " + str(avg_ep_rews_1) + " , " + str(avg_ep_rews_2) + " , " + str(round(self.ep_rew_best, 3)) + " , " + str(round(self.R_1_best,2)) + " , " + str(round(self.R_2_best,2)) + " , " + str(avg_actor_loss) + " , " + str(avg_critic_loss) + " , " + str(round(self.R_1_batch,2)) + " , " + str(round(self.R_1_sum,2)) + " , " + str(round(self.R_1_avg,2)) + " , " + str(round(self.R_2_batch,2)) + " , " + str(round(self.R_2_sum,2)) + " , " + str(round(self.R_2_avg,2)) + " , " + str(self.actor_1_lr) + " , " + str(self.critic_1_lr)  + " , " + str(self.actor_2_lr) + " , " + str(self.critic_2_lr) + "\n")
    rl_log_file.close()

    # Update lr scheduler 
    #self.actor_schedule.step(avg_ep_rews
    #self.critic_schedule.step(avg_ep_rews)
      
    # Reset batch-specific logging data
    self.logger['batch_lens'] = []
    self.logger['batch_rews_1'] = []
    self.logger['batch_rews_2'] = []
    self.logger['actor_losses_1'] = []
    self.logger['actor_losses_2'] = []
    self.logger['critic_losses_1'] = []
    self.logger['critic_losses_2'] = []
    
model = MAA2C()
model.learn(3000)

