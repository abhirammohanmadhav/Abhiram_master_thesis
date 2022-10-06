import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN_crit_2(nn.Module):

  # Define layers
  def __init__(self, in_dim, out_dim):

    super(FeedForwardNN_crit_2, self).__init__()

    self.layer1 = nn.Linear(in_dim, 64)
    self.layer2 = nn.Linear(64, 64)
    self.layer3 = nn.Linear(64, out_dim)

  #Define activation functions
  def forward(self, obs):
    #c = np.size(obs_2)
    #d = np.size(obs_1)
    # Convert observation to tensor if it's a numpy array
    if isinstance(obs, np.ndarray):
      obs = torch.tensor(obs, dtype=torch.float)
  
    #if isinstance(obs_2, np.ndarray):
      #obs_2 = torch.tensor(obs_2, dtype=torch.float)
    #obs_1 = torch.multiply(obs_1, 0.5)

    #obs = torch.empty((1, 4), dtype=torch.float)
    #obs[0,0:2] = obs_2
    #obs[0,2:4] = obs_1
    #obs = torch.cat((obs_2, obs_1), 0)
    activation1 = F.leaky_relu(self.layer1(obs),0.01)
    #activation1 = torch.cat([obs_2, obs_1], 1)
    activation2 = F.leaky_relu(self.layer2(activation1),0.01)
    output = self.layer3(activation2)

    return output   
