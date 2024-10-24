import torch
from torch.utils.data import Dataset

class PlaceholderDataset(Dataset):
    
    def __init__(self, param_a, param_b):
        
        self.param_a = param_a
        self.param_b = param_b
        
    def __len__(self):
        return 1

    def __getitem__(self, idx : int):
        
        return torch.Tensor([self.param_a, self.param_b])