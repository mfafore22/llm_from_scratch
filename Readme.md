# LLM From Scratch

A complete large language model built from the ground up in raw PyTorch — no
high-level libraries hiding the internals. Every component, from the attention
mechanism to the training loops, is implemented by hand to understand *how* and
*why* each piece works.

This is a learning-by-building project: the goal is a working pipeline that
covers the full lifecycle of a modern LLM, not a toy demo.

## Why this exists

Most people use LLMs through an API or a library like Hugging Face, where the
hard parts are abstracted away. This project takes the opposite path — building
each layer manually — to make the architecture and training process fully
transparent and reproducible.

## Scope

The project aims to cover the entire modern LLM pipeline:

| Stage | What it does |
|-------|--------------|
| **Architecture** | The transformer itself: attention, feed-forward layers, embeddings, normalization, the blocks that combine them |
| **Pretraining** | Training the base model on raw text to predict the next token |
| **SFT** | Supervised fine-tuning on instruction/response pairs to make the model follow directions |
| **Reward modeling** | Training a model to score responses by quality |
| **DPO / PPO** | Preference optimization — aligning the model with human preferences |
| **GRPO-style reasoning** | Reinforcement-style training to improve step-by-step reasoning |

## Status

Early stage — building the architecture component by component, bottom up.

- [x] **Multi-head attention** — the core mechanism that lets each token attend
      to every other token
- [ ] Position-wise feed-forward network
- [ ] Layer normalization & residual connections
- [ ] Transformer block (attention + FFN + residuals)
- [ ] Positional encoding
- [ ] Full model (stacked blocks + embeddings + output head)
- [ ] Tokenizer
- [ ] Pretraining loop
- [ ] SFT
- [ ] Reward model
- [ ] DPO / PPO
- [ ] GRPO reasoning

## Components

### Multi-Head Attention

The attention mechanism is the heart of a transformer. It lets each token in a
sequence look at every other token and decide how much to focus on each one,
building a context-aware representation.

"Multi-head" means this looking-around process runs several times in parallel,
with each head free to focus on a different kind of relationship (grammar,
meaning, position, etc.). The results are then combined.

Each token produces three vectors:

- **Query** — "what am I looking for?"
- **Key** — "what do I offer?"
- **Value** — "the information I carry"

Tokens are scored by matching Queries against Keys; those scores become weights
used to blend the Values together. The output has the same shape as the input,
which is what allows these blocks to be stacked.

**File:** `multiheadattention.py`

```python
from multiheadattention import MultiHeadAttention
import torch

attention = MultiHeadAttention(embedding_dim=512, number_of_heads=8)

# (batch_size, sequence_length, embedding_dim)
tokens = torch.randn(2, 10, 512)

# self-attention: same input is used for Query, Key, and Value
output = attention(tokens, tokens, tokens)

print(output.shape)  # torch.Size([2, 10, 512]) — same shape in, same shape out
```

## Requirements

- Python 3.10+
- PyTorch

## Setup

Using `uv`:

```bash
uv venv
uv add torch
```

Or with pip:

```bash
pip install torch
```

## Reference

The architecture follows the transformer introduced in *Attention Is All You
Need* (Vaswani et al., 2017), implemented from scratch rather than imported.