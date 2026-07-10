# The full transformer: ties every piece together into one model.
# Flow: turn words into numbers → stamp positions → run down the encoder stack →
# run down the decoder stack (consulting the encoder) → score every possible
# next word → out.

import torch
import torch.nn as nn

from positionalencoding import PositionalEncoding
from encoderlayer import EncoderLayer
from decoderlayer import DecoderLayer


class Transformer(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, embedding_dim, num_heads,
                 num_layers, hidden_dim, max_seq_length, dropout=0.1):
        # source_vocab_size: how many distinct words the input language has
        # target_vocab_size: how many distinct words the output language has
        # embedding_dim:     size of each word's vector (must match across all pieces)
        # num_heads:         how many parallel attention passes
        # num_layers:        how many encoder/decoder blocks to stack
        # hidden_dim:        feed-forward workspace size (usually 4 × embedding_dim)
        # max_seq_length:    longest sentence supported
        # dropout:           fraction of values randomly zeroed during training
        super(Transformer, self).__init__()

        # Word-to-numbers translators (lookup tables)
        self.encoder_embedding = nn.Embedding(source_vocab_size, embedding_dim)
        self.decoder_embedding = nn.Embedding(target_vocab_size, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_seq_length)

        # Stack of encoder blocks
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Stack of decoder blocks
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embedding_dim, num_heads, hidden_dim, dropout)
            for _ in range(num_layers)
        ])

        # Final "answer" machine: scores every possible output word
        self.output_projection = nn.Linear(embedding_dim, target_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, source, target):
        # Blindfold 1: ignore padding (0s) in the source
        source_mask = (source != 0).unsqueeze(1).unsqueeze(2)

        # Blindfold 2a: ignore padding in the target
        target_mask = (target != 0).unsqueeze(1).unsqueeze(3)

        # Blindfold 2b: "no peeking ahead" — hide future positions
        seq_length = target.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()

        # Combine both target rules: ignore padding AND don't peek ahead
        target_mask = target_mask & nopeak_mask
        return source_mask, target_mask

    def forward(self, source, target):
        source_mask, target_mask = self.generate_mask(source, target)

        # Encoder: words → numbers → positions stamped → down the stack
        source_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(source)))
        encoder_output = source_embedded
        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output, source_mask)

        # Decoder: same prep, then down the stack while consulting the encoder
        target_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(target)))
        decoder_output = target_embedded
        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output, encoder_output, source_mask, target_mask)

        # Final projection: score every possible next word
        output = self.output_projection(decoder_output)
        return output