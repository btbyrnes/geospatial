import torch
import torch.nn as nn
import torch.nn.functional as F


class geoModel(nn.Module):


    def __init__(self, points, embedding_dimension, context_size, intermediate_size=512):

        super(geoModel, self).__init__()
        
        self.embeddingLayer = nn.Embedding(points, embedding_dimension)
        self.linear1 = nn.Linear(embedding_dimension*context_size, 128)
        self.linear2 = nn.Linear(128, points)
  

    def forward(self, inputs):
        
        input_size = inputs.shape[0]

        embeddings = self.embeddingLayer(inputs).view((input_size,-1))
        output = F.relu(self.linear1(embeddings))
        log_probs = F.log_softmax(self.linear2(output),dim=1)

        return log_probs


    def predict(self, inputs):

        with torch.no_grad():
            
            return self.forward(inputs)


