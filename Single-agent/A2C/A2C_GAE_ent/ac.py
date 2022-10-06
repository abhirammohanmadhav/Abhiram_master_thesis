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
from geometry_generator import generate_geometry_init
from geometry_generator import generate_geometry_train

from os import path

class AC:

  def __init__(self):

    self.obs_dim = 3 # [bump_radius, delta p, delta T]

    # Call hyperparameters
    self._init_hyperparameters()
    
    # Initialize actor and critic networks
    self.actor = FeedForwardNN_act(self.obs_dim, self.act_dim)
    self.critic = FeedForwardNN_crit(self.obs_dim, 1)

    # Initialize optimizers for actor and critic
    self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
    self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    # If restart, load actor and critic models
    if self.restart == 1:
  
      self.actor.load_state_dict(torch.load('./ac_actor.pth'))
      self.actor_optim.load_state_dict(torch.load('./ac_actor_opt.pth'))
      self.critic.load_state_dict(torch.load('./ac_critic.pth'))
      self.critic_optim.load_state_dict(torch.load('./ac_critic_opt.pth'))
    
    # Else init log file if new start
    elif self.restart == 0:
    
       rl_log_file = open("rl.log", "w") #write to file without line
       rl_log_file.write("avg_ep_lens , avg_ep_rews , avg_actor_loss , self.t_so_far , self.i_so_far \n")
       rl_log_file.close()

       reward_log_file = open("reward.log", "w") #write to file without line
       reward_log_file.write("self.i_so_far , t , center , radius , p_33, p , |((p_inl-p_0)/(p_33-p_0))-1| , T_33, T , (T_outl-T_0)/(T_33-T_0) , rew_delta, rew \n")
       reward_log_file.close()


    self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
  
    self.cov_mat = torch.diag(self.cov_var)
    
    # Logger to monitor progress
    self.logger = {
           't_so_far':0,          # time steps so far
           'i_so_far':0,          # iterations so far
           'batch_lens': [],      # episodic lengths in batch
           'batch_rews': [],      # episodic returns in batch
           'actor_losses': [],    #losses of actor network in current iteration
      }

    # Generate domain with bump at min radius
    r_min=generate_geometry_start(self.bump_center,self.bump_radius_min,self.bump_radius_min,self.bump_radius_max)

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
    r_max=generate_geometry_start(self.bump_center,self.bump_radius_max,self.bump_radius_min,self.bump_radius_max)
     
    # Run simulation with bump at max radius
    os.system("$MPIEXEC $FLAGS_MPI_BATCH /hpcwork/jara0203/RL/MAIA/Solver/src/maia properties_run.toml > maia_command.log")

    # Receive feedback from MAIA 
    max_p_inl_norm,max_T_outl_norm,max_p_inl,max_T_outl = self.response_from_maia()
   
    # Set new extrema for p_inl and T_outl
    self.p_33=max_p_inl
    self.T_33=max_T_outl
       
    #Move output files
    os.system("mv Output_SegNo_0_BC_1022.dat output_files/max_Output_SegNo_0_BC_1022.dat")
    os.system("mv Output_SegNo_1_BC_4130.dat output_files/max_Output_SegNo_1_BC_4130.dat")
   

  def learn(self, total_timesteps):

    print(f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ", end='')
    print(f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps")

    if self.restart == 0:

       self.t_so_far = 0 # Timesteps simulated so far
       self.i_so_far = 0 # Iterations ran so far
    
    elif self.restart == 1:

       time_steps_before_restart = np.loadtxt('rl.log', delimiter ="," , usecols =[3], skiprows=1)
       iterations_before_restart = np.loadtxt('rl.log', delimiter ="," , usecols =[4], skiprows=1)
        
       self.t_so_far = int(time_steps_before_restart[time_steps_before_restart.shape[0]-1]) # Timesteps simulated so far
       self.i_so_far = int(iterations_before_restart[iterations_before_restart.shape[0]-1]) # Iterations ran so far
      
    #Initialize observation array  
    obs = np.zeros((self.obs_dim)) 

    # Overall loop
    while self.t_so_far < total_timesteps:    
    
      # Perform one batch with "time_steps_per_batch" time steps  
      batch_obs, batch_acts, batch_log_probs, batch_rtgs,batch_advs, batch_lens = self.rollout(obs)

      # Calculate how many timesteps we collected this batch   
      self.t_so_far += np.sum(batch_lens)

      # Increment the number of iterations
      self.i_so_far += 1

      # Logging timesteps so far and iterations so far
      self.logger['t_so_far'] = self.t_so_far
      self.logger['i_so_far'] = self.i_so_far

      # Calculate V_{phi, k}
      V, _, _ = self.evaluate(batch_obs, batch_acts)
      
      for _ in range(self.n_updates_per_iteration):

        # Calculate V_phi and pi_theta(a_t | s_t)
        V, curr_log_probs, dist_entropies = self.evaluate(batch_obs, batch_acts)
        # actor and critic losses
        actor_loss = -torch.multiply(curr_log_probs, batch_advs).mean()
        
        actor_loss = actor_loss - dist_entropies * self.entropy_beta
        critic_loss = nn.MSELoss()(V, batch_rtgs)        

        # Calculate gradients and perform backward propagation for actor network
        self.actor_optim.zero_grad()
        actor_loss.backward(retain_graph=True) 
        self.actor_optim.step()

        # Calculate gradients and perform backward propagation for critic network    
        self.critic_optim.zero_grad()    
        critic_loss.backward()    
        self.critic_optim.step()

        # Log actor loss
        self.logger['actor_losses'].append(actor_loss.detach())

      # Print a summary of our training so far
      self._log_summary()

      # Save our model if it's time
      if self.i_so_far % self.save_freq == 0:
        
        torch.save(self.actor.state_dict(), './ac_actor.pth')
        torch.save(self.actor_optim.state_dict(), './ac_actor_opt.pth')
        torch.save(self.critic.state_dict(), './ac_critic.pth')
        torch.save(self.critic_optim.state_dict(), './ac_critic_opt.pth')
      
      # Make backups if it's time
      if self.i_so_far % self.backup_freq == 0:
        
        torch.save(self.actor.state_dict(), './ac_actor_'+'%06d'%self.i_so_far+'.pth')
        torch.save(self.actor_optim.state_dict(), './ac_actor_opt_'+'%06d'%self.i_so_far+'.pth')
        torch.save(self.critic.state_dict(), './ac_critic_'+'%06d'%self.i_so_far+'.pth')
        torch.save(self.critic_optim.state_dict(), './ac_critic_opt_'+'%06d'%self.i_so_far+'.pth')
  
  # Collect data for a batch
  def rollout(self,obs):

    # Initialize lists
    batch_obs = []             # batch observations [#ts per batch, dim of obs]
    batch_acts = []            # batch actions [#ts per batch, dim of action]
    batch_vals = []
    batch_advs = []
    batch_log_probs = []       # log probs of each action [#ts per batch]
    batch_rews = []            # batch rewards [#episodes, #ts per episode]
    batch_rtgs = []            # batch rewards-to-go [#ts per batch]
    batch_lens = []            # episodic lengths in batch [#episodes]

    # Number of timesteps run so far this batch
    t = 0

    # Keep simulating until we've run more than or equal to specified timesteps per batch
    while t < self.timesteps_per_batch:
  
      # Rewards this episode
      ep_rews = []
      ep_vals = []
  
      # Update properties_restart.toml
      ts = 1
      self.reset_restart_property_file()

      # Generate domain with initial bump
      obs[0] = generate_geometry_start(self.bump_center,self.bump_radius_init,self.bump_radius_min,self.bump_radius_max)
      
      # Run simulation with initial bump
      os.system("$MPIEXEC $FLAGS_MPI_BATCH /hpcwork/jara0203/RL/MAIA/Solver/src/maia properties_run.toml > maia_command.log")
      
      # get response from MAIA
      obs[1],obs[2],p,T = self.response_from_maia()
      
      #Move output files
      os.system("rm Output_SegNo_0_BC_1022.dat")
      os.system("rm Output_SegNo_1_BC_4130.dat")

      #Init flag
      done = False

      # Run an episode for a maximum of max_time_steps_per_episode timesteps
      for ep_t in range(self.max_timesteps_per_episode):

        # Increment timesteps ran this batch so far
        t += 1
    
        # Collect observation 
        batch_obs.append(obs)
        action, log_prob = self.get_action(obs) 
        action_tensor = torch.tensor(action, dtype=torch.float)
        V, _, _ = self.evaluate(obs,action_tensor)
        radius_change=action*3
        
        # Generate new domain
        obs[0],death = generate_geometry_train(self.i_so_far,t,self.bump_center,obs[0],radius_change,self.bump_radius_min,self.bump_radius_max)
        # Run simulation
        os.system("$MPIEXEC $FLAGS_MPI_BATCH /hpcwork/jara0203/RL/MAIA/Solver/src/maia properties_restart.toml > maia_command.log")
        
        # Update properties_restart.toml
        self.update_restart_property_file(ts)
        ts += 1

        #Receive response (delta p, delta t)  
        obs[1],obs[2],p,T = self.response_from_maia()
        
        #Move output files 
        os.system("rm Output_SegNo_0_BC_1022.dat")
        os.system("rm Output_SegNo_1_BC_4130.dat")
        
         
        rew=0
        rew_p=np.abs(obs[1]-1) #pressure
        rew_T=obs[2] #Temperature       
        rew_delta=np.abs(rew_p-rew_T) 
        rew=2-rew_delta 

        #Scale reward
        rew=rew*rew*rew*rew*rew #Max=32
        
        #Normalize reward
        rew=rew/32

        #If agent reaches target, stop and give high reward
        if rew_delta <=0.1:
           done=True
 
        #Denormalize radius
        radius_denorm=obs[0]*(self.bump_radius_max-self.bump_radius_min)+self.bump_radius_min

        #Punish with 0 reward of agent exceeds R_max or R_min
        if death==True:
           rew=0
           done=True

        #Write to reward.log file
        reward_log_file = open("reward.log", "a") #write to file without line
        reward_log_file.write(str(self.i_so_far) + " , " + str(t) + " , " + str(self.bump_center) + " , " + str(radius_denorm) + " , " + str(self.p_33) + " , " + str(p) + " , " +str(rew_p) + " , " + str(self.T_33) + " , " + str(T) + " , " + str(rew_T) + " , " + str(rew_delta) + " , " + str(rew) + "\n")
        reward_log_file.close()

        #Define when task is done --> Later version

        # Collect reward, action, and log prob
        ep_rews.append(rew)
        ep_vals.append(V)
        batch_acts.append(action)
        batch_log_probs.append(log_prob)
    
        # If done = True, meaning if a task is done, break.
        if done:
          t=t+self.max_timesteps_per_episode-(ts-1)    
          break

      # Collect episodic length and rewards
      batch_lens.append(ep_t + 1) 
      batch_rews.append(ep_rews)
      batch_vals.append(ep_vals)

    # Reshape data as tensors in the shape specified before returning
    batch_obs = torch.tensor(batch_obs, dtype=torch.float)
    batch_acts = torch.tensor(batch_acts, dtype=torch.float)
    batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
    
    # Compute rewards to go
    batch_rtgs, batch_advs = self.compute_advantage(batch_rews, batch_vals)
    
    # Log the episodic returns and episodic lengths in this batch.
    self.logger['batch_rews'] = batch_rews
    self.logger['batch_lens'] = batch_lens

    # Return the batch data
    return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_advs, batch_lens

  def compute_advantage(self, batch_rews, batch_vals):
  
    if self.gae==False:
       batch_rtgs = []

       # Iterate through each episode backwards
       for ep_rews in reversed(batch_rews):
    
         discounted_reward = 0 # The discounted reward so far

         for rew in reversed(ep_rews):
    
           discounted_reward = rew + discounted_reward * self.gamma
           batch_rtgs.insert(0, discounted_reward)
       
       batch_rtgs = (batch_rtgs - batch_rtgs.mean()) / (batch_rtgs.std() + 1e-10)
    # If GAE, ...
    elif self.gae==True:
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
          
    # Convert the rewards-to-go into a tensor
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
    # List for batch values in correct order    
    batch_vals_list = []    
    
    # Iterate through each episode and store as tensor
    for ep_vals in(batch_vals):
  
       for val in(ep_vals):

          batch_vals_list.append(val)      

    batch_vals = torch.tensor(batch_vals_list, dtype=torch.float)

    # Calculate advantage
    A = batch_rtgs - batch_vals.detach() 

    # Normalize advantages
    A = (A - A.mean()) / (A.std() + 1e-10)
    batch_rtgs = (batch_rtgs - batch_rtgs.mean()) / (batch_rtgs.std() + 1e-10)
    return batch_rtgs, A

  def get_action(self, obs):
    mean = self.actor(obs)
    # Create our Multivariate Normal Distribution
    dist = MultivariateNormal(mean, self.cov_mat)
    # Sample an action from the distribution and get its log prob
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.detach().numpy(), log_prob.detach() 

  def evaluate(self, batch_obs, batch_acts):
    V = self.critic(batch_obs).squeeze() 
    mean = self.actor(batch_obs)
    dist = MultivariateNormal(mean, self.cov_mat)
    log_probs = dist.log_prob(batch_acts)
    dist_entropy = dist.entropy().mean()
    return V, log_probs, dist_entropy

  #Define hyperparameters
  def _init_hyperparameters(self):
    self.timesteps_per_batch = 30  #4800
    self.max_timesteps_per_episode = 10 #1600
    self.gamma = 0.9 # Discount factor As recommended by the paper
    self.lam = 0.95
    self.gae = True 
    self.n_updates_per_iteration = 5 #usually between 5 and 20
    self.entropy_beta = 0.01
    self.clip = 0.2 # Clipping parameter for trust region As recommended by the paper
    self.lr = 0.005 #Learning rate
    self.save_freq = 1 # How often we save in number of iterations
    self.backup_freq = 10 # How often we backup in number of iterations
    self.restart = 1 #Restart or not

    #Bump parameters
    self.bump_center = 100
    self.bump_radius_init = 10
    self.bump_radius_min = 0
    self.bump_radius_max = 33

    # We need to define the MAIA time step in here
    self.maia_time_steps = 150000
    self.maia_init_time_steps = 300000

    # Reward function (dummy init values that will be overwritten)
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
    self.t_so_far = self.logger['t_so_far']
    self.i_so_far = self.logger['i_so_far']
    avg_ep_lens = np.mean(self.logger['batch_lens'])
    avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
    avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
    avg_ep_lens = str(round(avg_ep_lens, 2))
    avg_ep_rews = str(round(avg_ep_rews, 2))
    avg_actor_loss = str(round(avg_actor_loss, 5))

    # Print logging statements
    print(flush=True)
    print(f"-------------------- Iteration #{self.i_so_far} --------------------", flush=True)
    print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
    print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
    print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
    print(f"Timesteps So Far: {self.t_so_far}", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)

    #Write to log file
    rl_log_file = open("rl.log", "a") 
    rl_log_file.write(str(avg_ep_lens) + " , " + str(avg_ep_rews) + " , " + str(avg_actor_loss) + " , " + str(self.t_so_far) + " , " + str(self.i_so_far) + "\n")
    rl_log_file.close()

    # Reset batch-specific logging data
    self.logger['batch_lens'] = []
    self.logger['batch_rews'] = []
    self.logger['actor_losses'] = []
    
model = AC()
model.learn(3000)

