import numpy as np
import torch
from torch.utils.data import Dataset

class geoDataSet(Dataset):


    def __init__(self, labels_file, features_file):
        
        super(geoDataSet).__init__()

        self.labels = torch.from_numpy(np.load(labels_file))
        self.features = torch.from_numpy(np.load(features_file))
        self.label_count = self.labels.shape[0]
        self.context_size = self.features.shape[1]


    def __len__(self):

        return len(self.labels)


    def __getitem__(self, idx):


        if torch.is_tensor(idx):
            idx = idx.to_list()

        X = self.features[idx]
        y = self.labels[idx]

        return {"context": X, "target": y}