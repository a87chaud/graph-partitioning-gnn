import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, num_partitions, dropout=0.3):
        super().__init__()
        
        self.layers = torch.nn.ModuleList()
        self.hidden_layers = torch.nn.ModuleList()
        self.dropout = dropout
        self.layers.append(SAGEConv(in_dim, hidden_dim))
        self.hidden_layers.append(torch.nn.BatchNorm1d(hidden_dim))
        # One more hidden layer
        self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.hidden_layers.append(torch.nn.BatchNorm1d(hidden_dim))
        
        self.layers.append(SAGEConv(hidden_dim, num_partitions))
    
    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers[:-1]):
            x = conv(x, edge_index)
            x = self.hidden_layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.layers[-1](x, edge_index)
        Y = F.softmax(x, dim=1)
        return Y