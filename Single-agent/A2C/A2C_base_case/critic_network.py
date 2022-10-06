import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN_crit(nn.Module):

  # Define layers
  def __init__(self, in_dim, act_dim, out_dim):

    super(FeedForwardNN_crit, self).__init__()

    self.layer1 = nn.Linear(in_dim, 64)
    self.layer2 = nn.Linear(64+act_dim, 64)
    self.layer3 = nn.Linear(64, out_dim)

  #Define activation functions
  def forward(self, obs, action):

    # Convert observation to tensor if it's a numpy array
    if isinstance(obs, np.ndarray) and isinstance(action, np.ndarray):
      obs = torch.tensor(obs, dtype=torch.float)
      action = torch.tensor(action, dtype=torch.float)
  
    activation1 = F.relu(self.layer1(obs))
    activation1 = torch.cat([activation1, action], 1)
    activation2 = F.relu(self.layer2(activation1))
    output = self.layer3(activation2)

    return output   
