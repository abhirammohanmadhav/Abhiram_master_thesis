import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN_crit(nn.Module):

  # Define layers
  def __init__(self, in_dim, out_dim):

    super(FeedForwardNN_crit, self).__init__()

    self.layer1 = nn.Linear(in_dim, 64)
    self.layer2 = nn.Linear(64, 64)
    self.layer3 = nn.Linear(64, out_dim)

  #Define activation functions
  def forward(self, obs):

    # Convert observation to tensor if it's a numpy array
    if isinstance(obs, np.ndarray):
      obs = torch.tensor(obs, dtype=torch.float)
  
    activation1 = F.leaky_relu(self.layer1(obs),0.01)
    activation2 = F.leaky_relu(self.layer2(activation1),0.01)
    output = self.layer3(activation2)

    return output   
