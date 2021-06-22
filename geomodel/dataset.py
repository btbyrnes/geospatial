import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class geoDataSet2(Dataset):

    def __init__(self, labels_file, features_file):
        
        super(geoDataSet).__init__()

        self.labels = np.load(labels_file)
        self.features = np.load(features_file)
        self.data = pd.DataFrame({"labels": self.labels, "1": self.features[:,0], "2": self.features[:,1], "3": self.features[:,2]})
        self.label_count = self.labels.shape[0]
        self.context_size = self.features.shape[1]


    def __len__(self):

        return len(self.labels)


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = torch.Tensor.tolist(idx)

        X = self.data.iloc[idx,1:]
        y = self.data.iloc[idx,0]

        X = torch.from_numpy(X.to_numpy())
        y = torch.from_numpy(np.array(y))

        return {"context": X, "target": y}


class geoDataSet(Dataset):


    def __init__(self, labels_file, features_file):
        
        super(geoDataSet).__init__()

        self.labels = torch.from_numpy(np.load(labels_file))
        self.features = torch.from_numpy(np.load(features_file))
        self.label_count = len(torch.unique(self.labels))
        self.context_size = self.features.shape[1]


    def __len__(self):

        return len(self.labels)


    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.to_list()

        X = self.features[idx]
        y = self.labels[idx]

        return {"context": X, "target": y}