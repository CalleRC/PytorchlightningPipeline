

from torch import nn
import torch

class PlaceholderModel(nn.Module):
    
    def __init__(self, param_a, param_b):
        
        self.param_a = param_a
        self.param_b = param_b
        
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return x