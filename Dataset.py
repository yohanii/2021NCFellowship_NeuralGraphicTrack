import numpy as np
import torch
from torch.utils.data import Dataset


class CMU_Dataset(Dataset):
    def __init__(self, x, y, device):
        self.n_sample = x.shape[0]
        self.n_input = x.shape[1]
        self.n_label = y.shape[1]

        self.x_data = torch.as_tensor(
            np.array(x).astype(np.float32)).to(device)
        self.y_data = torch.as_tensor(
            np.array(y).astype(np.float32)).to(device)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_sample

    def getInputSize(self):
        return self.n_input

    def getOutputSize(self):
        return self.n_label
