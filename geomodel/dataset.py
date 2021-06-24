import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class geoDataSet(Dataset):

    def __init__(self, data_file):
        
        super(geoDataSet).__init__()

        data = np.load(data_file)

        self.X = torch.from_numpy(data[:,1:])
        self.Y = torch.from_numpy(data[:,0])
        self.len = len(self.Y)
        

    def __len__(self):

        return self.len


    def __getitem__(self, idx):

        X = self.X[idx]
        Y = self.Y[idx]

        return {"context": X, "target": Y}

