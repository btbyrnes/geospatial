import torch
import torch.nn as nn
import torch.nn.functional as F


class geoModel(nn.Module):


    def __init__(self, number_of_points, embedding_dimension, feature_size, intermediate_size=128):

        super(geoModel, self).__init__()
        
        self.embeddingLayer = nn.Embedding(number_of_points, embedding_dimension)
        print(self.embeddingLayer)
        self.linear1 = nn.Linear(feature_size*embedding_dimension, intermediate_size)
        print(self.linear1)
        self.linear2 = nn.Linear(intermediate_size, number_of_points)
  

    def forward(self, inputs):

        if inputs.dim() > 1:
            input_rows = inputs.shape[inputs.dim()-2]
        else:
            input_rows = 1

        embeddings = self.embeddingLayer(inputs).view([input_rows,-1])
        output = F.relu(self.linear1(embeddings))
        log_probs = F.log_softmax(self.linear2(output),dim=-1)

        return log_probs


    def predict(self, inputs):

        if inputs.dim() == 1:
            inputs = inputs.reshape(1,-1)

        with torch.no_grad():
            preds = self.forward(inputs)
            points = torch.argmax(preds,1)
            
            return points