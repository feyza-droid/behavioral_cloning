import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv("data/" + csv_file + ".csv")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = self.data.iloc[idx, 1:42]
        state = np.array(state)
        action = self.data.iloc[idx, 42:44]
        action = np.array(action)
        sample = torch.from_numpy(state).float(), torch.from_numpy(action).float()

        return sample
        
#dataset = MyDataset('idm')
#print(dataset[1])