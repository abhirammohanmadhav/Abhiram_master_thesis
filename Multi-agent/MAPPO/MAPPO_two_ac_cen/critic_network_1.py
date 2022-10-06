import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN_crit_1(nn.Module):

  # Define layers
  def __init__(self, in_dim, out_dim):

    super(FeedForwardNN_crit_1, self).__init__()

    self.layer1 = nn.Linear(in_dim, 64)
    self.layer2 = nn.Linear(64, 64)
    self.layer3 = nn.Linear(64, out_dim)

  #Define activation functions
  def forward(self, obs):
    #a = np.size(obs_1)
    #b = np.size(obs_2)
    # Convert observation to tensor if it's a numpy array
    if isinstance(obs, np.ndarray):
      obs = torch.tensor(obs, dtype=torch.float)
  
    #if isinstance(obs_2, np.ndarray):
      #obs_2 = torch.tensor(obs_2, dtype=torch.float)
    #obs_2 = torch.multiply(obs_2, 0.5)
    #obs = torch.empty((1,4), dtype=torch.float)
    #print('shape of empty tensor ', obs.size())
    #obs[0,0:2] = obs_1
    #print('shape of half filled tensor ', obs.size())
    #obs[0,2:4] = obs_2
    #obs_11 = torch.cat([obs_1, obs_2], 0).flatten()
    #print('shape of obs_11', obs_11.size(), obs_11)
    #obs = torch.cat([obs_11, obs_e], 0)
    #print(obs.size())
    #obs_11 = torch.reshape(obs_11, (4, 4))
    activation1 = F.leaky_relu(self.layer1(obs),0.01)
    #print(activation1.size())
    #activation1 = torch.cat([activation1, obs_2], 1)
    activation2 = F.leaky_relu(self.layer2(activation1),0.01)
    output = self.layer3(activation2)

    return output   
