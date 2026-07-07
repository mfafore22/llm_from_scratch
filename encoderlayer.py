#This layer combines multi head attention with the 
#the feed-forward network , using reisdual connections
#and layer normalization for stable training

import torch 
import torch.nn as nn

from multiheadattention import MultiHeadAttention
from feedforward import FeedForward

class EncoderLayer(nn.Module):
    def __init__(self,embedding_dim, num_heads, hidden_dim, dropout=0.1 ):
        #embedding_dim: size of each word's vector
        #num_heads: how many parallel attention passes
        #hidden_dim: feed-forward workspace size 
        #dropout: fraction of values randomly zeroes during training
        super(EncoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.feed_forward = FeedForward(embedding_dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(embedding_dim) #Tidy up after attention
        self.norm2 = nn.LayerNorm(embedding_dim) #Tidy up after feedforward
        self.dropout = nn.Dropout(dropout) #shake up tool

    def forward(self,x , mask=None):
        #Self attention with residual connection and layer norm
        attention_output = self.self_attention(x, x, x, mask) # run attention, keep result aside
        x = self.norm1(x + self.dropout(attention_output)) # add original back , then tidy

        #Feed-forward with residual connection and layer norm
        feed_forward_output = self.feed_forward(x) #run tune-up, keep result aside
        x = self.norm2(x + self.dropout(feed_forward_output))

        return x 

