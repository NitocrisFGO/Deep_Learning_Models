import torch
import torch.nn as nn

embed_dim = 256
num_heads = 8
multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

# Example input tensor
x = torch.rand(10, 32, embed_dim)  # (sequence_length, batch_size, embed_dim)

# Self-attention
attn_output, attn_output_weights = multihead_attn(x, x, x)
print(attn_output.shape)
print(attn_output_weights.shape)
