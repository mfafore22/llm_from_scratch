import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, number_of_heads):
        #embedding_dim : size of each token's vector
        #number_of_heads: how many parallel attention "heads to run"
        super(MultiHeadAttention, self).__init__()

        #The embedding must split evenly across the heads or the math breaks
        assert embedding_dim % number_of_heads == 0

        self.embedding_dim = embedding_dim
        self.number_of_heads = number_of_heads
        self.head_dim = embedding_dim // number_of_heads #slice of the vector each heads works on 

        #Learnable projections that turn the input into Query , Key and value
        self.query_projection = nn.Linear(embedding_dim, embedding_dim)
        self.key_projection = nn.Linear(embedding_dim, embedding_dim)
        self.value_projection = nn.Linear(embedding_dim, embedding_dim)

        # Final projection that mizes all heads results back together
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def scaled_dot_product_attention(self, queries, keys , values, mask=None):
        #Score how much each token should attend to every other token
        #Queries @ keys^T compare every Query against every key
        #Dividing by sqrt(head_dim) keeps the number from blowing up
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Turn raw scores into probabilities that sum to 1 across each row.
        attention_weights = F.softmax(attention_scores, dim=1)

        #Blend the values together using those probabilities as weights
        weighted_values = torch.matmul(attention_weights, values)

        return weighted_values, attention_weights
    
    def forward(self, query_input, key_input, value_input , mask=None):
        batch_size = query_input.size(0)

        queries = self.query_projection(query_input).view(batch_size, -1, self.number_of_heads, self.head_dim).transpose(1,2)

        keys = self.key_projection(key_input).view(batch_size, -1 ,self.number_of_heads, self.head_dim).transpose(1,2)

        values = self.value_projection(value_input).view(batch_size, -1, self.number_of_heads, self.head_dim ).transpose(1,2)

        #Run scaled dot-product attention on every head at once.
        attention_output , attention_Weights = self.scaled_dot_product_attention(queries, keys, values, mask)

        attention_output = attention_output.transpose(1,2).contiguous().view(batch_size, -1, self.embedding_dim)

        #One last projection lets the heads results combine and interact.
        return self.output_projection(attention_output)


mha = MultiHeadAttention(embedding_dim=512, number_of_heads=8)
