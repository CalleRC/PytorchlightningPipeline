import torch
from torch.utils.data import Dataset

class PlaceholderDataset(Dataset):
    
    def __init__(self, param_a, param_b, is_train = True, **kwargs):
        
        self.param_a = param_a
        self.param_b = param_b
        
        self.is_train = is_train
        
    def __len__(self):
        return 1

    def __getitem__(self, idx : int):
        
        return torch.Tensor([self.param_a, self.param_b])