#After the attention mechanim helps the model understand the relationships
#between words , each word is then processed individually .Think of it like 
#giving each word its own quick "upgrade". The model adjusts the word's information
#filters out less useful parts and finetunes its before sending it to the next layer
#This helps the model create a richer and more meaningful representation of each word.

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, embedding_dim,hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.expand= nn.Linear(embedding_dim, hidden_dim)
        self.contract = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        #Get the packet expand it . judge it , shake it and shrink it then give it back
        x = self.expand(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.contract(x)
        return x
