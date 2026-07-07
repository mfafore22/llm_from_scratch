# A single decoder block. Same skeleton as the encoder block, but with one extra
# attention station in the middle and a "no peeking" mask on the first one:
#   1. masked self-attention  — look at my own words-so-far, but never peek ahead
#   2. encoder-decoder attention — consult the encoder's understanding of the source
#   3. feed-forward — per-word tune-up
# Each station is wrapped with a residual connection (add the original back) and
# layer normalization (tidy up the numbers).

import torch
import torch.nn as nn

from multiheadattention import MultiHeadAttention
from feedforward import FeedForward


class DecoderLayer(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout=0.1):
        # embedding_dim: size of each word's vector (must match across all pieces)
        # num_heads:     how many parallel attention passes
        # hidden_dim:    feed-forward workspace size (usually 4 × embedding_dim)
        # dropout:       fraction of values randomly zeroed during training
        super(DecoderLayer, self).__init__()

        self.self_attention = MultiHeadAttention(embedding_dim, num_heads)      # look at my own words
        self.encoder_attention = MultiHeadAttention(embedding_dim, num_heads)   # look at encoder's output
        self.feed_forward = FeedForward(embedding_dim, hidden_dim, dropout)     # per-word tune-up
        self.norm1 = nn.LayerNorm(embedding_dim)                               # tidy-up after masked attention
        self.norm2 = nn.LayerNorm(embedding_dim)                               # tidy-up after encoder attention
        self.norm3 = nn.LayerNorm(embedding_dim)                               # tidy-up after feed-forward
        self.dropout = nn.Dropout(dropout)                                    # shake-up tool

    def forward(self, x, encoder_output, source_mask=None, target_mask=None):
        # Masked self-attention — look at my own words-so-far, blindfolded from future words
        attention_output = self.self_attention(x, x, x, target_mask)
        x = self.norm1(x + self.dropout(attention_output))

        # Encoder-decoder attention — question comes from x, material comes from the encoder
        attention_output = self.encoder_attention(x, encoder_output, encoder_output, source_mask)
        x = self.norm2(x + self.dropout(attention_output))

        # Feed-forward network — per-word tune-up
        feed_forward_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(feed_forward_output))

        return x