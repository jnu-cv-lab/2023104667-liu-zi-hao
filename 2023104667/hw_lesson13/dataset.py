import numpy as np
import torch
from torch.utils.data import Dataset

class BadmintonDataset(Dataset):
    def __init__(self, data_path, label_path):
        self.data = np.load(data_path).astype(np.float32)
        self.label = np.load(label_path).astype(np.int64)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        y = torch.tensor(self.label[idx])
        return x, y