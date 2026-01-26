# -*- coding: utf-8 -*-

# @ Moritz Rempe, moritz.rempe@uk-essen.de
# Institute for Artifical Intelligence in Medicine,
# University Medicine Essen

import torch
import torch.nn as nn
from model.complex_layer.activations import ComplexGeLU
import model.complex_layer.spectral_layer as S


class ComplexMultiHeadAttention(nn.Module):
    """
    Complex-valued Multi-Head Attention that supports both self-attention
    and cross-attention by accepting separate query, key, and value inputs.
    """

    def __init__(self, embed_dim, num_heads):
        super(ComplexMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        # Linear layers to project input to Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=torch.cfloat)
        self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=torch.cfloat)
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=torch.cfloat)

        # Final output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, dtype=torch.cfloat)

    def forward(self, query, key, value, return_attention=False):
        """
        Forward pass for multi-head attention.

        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, query_len, embed_dim).
            key (torch.Tensor): Key tensor of shape (batch_size, key_len, embed_dim).
            value (torch.Tensor): Value tensor of shape (batch_size, key_len, embed_dim).
            return_attention (bool): Whether to return the attention weights.

        Returns:
            A tuple of (output, attention_weights) if return_attention is True,
            otherwise just the output tensor.
        """
        batch_size = query.shape[0]
        query_len = query.shape[1]
        key_len = key.shape[1]

        # Project query, key, and value from their respective sources
        q = self.q_proj(query)  # (batch_size, query_len, embed_dim)
        k = self.k_proj(key)  # (batch_size, key_len, embed_dim)
        v = self.v_proj(value)  # (batch_size, key_len, embed_dim)

        # Reshape for multi-head attention
        # (batch_size, seq_len, num_heads, head_dim) -> (batch_size, num_heads, seq_len, head_dim)
        q = q.view(batch_size, query_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, key_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention scores using the Hermitian inner product (q * k^H)
        # scores shape: (batch_size, num_heads, query_len, key_len)
        scores = torch.matmul(q, k.transpose(-2, -1).conj()) / self.head_dim**0.5

        # --- Your chosen complex softmax implementation ---
        # This applies softmax to the real and imaginary parts independently.
        attn_weights_real = nn.functional.softmax(scores.real, dim=-1)
        attn_weights_imag = nn.functional.softmax(scores.imag, dim=-1)
        # Note: Taking .abs() here combines them but loses phase information of the weights.
        # This might be desired. If not, you could use the complex weights directly.
        attention_weights = torch.complex(attn_weights_real, attn_weights_imag)

        # Apply attention to the value tensor
        # Here we use the separate real/imag weights on the real/imag parts of v
        context_real = torch.matmul(attn_weights_real, v.real)
        context_imag = torch.matmul(attn_weights_imag, v.imag)
        attention_output = torch.complex(context_real, context_imag)
        # --- End of complex softmax implementation ---

        # Concatenate heads and project
        # (batch_size, num_heads, query_len, head_dim) -> (batch_size, query_len, embed_dim)
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, query_len, self.embed_dim)
        )
        attention_output = self.out_proj(attention_output)

        if return_attention:
            # Return absolute value of weights for visualization purposes
            return attention_output, attention_weights.abs()
        else:
            return attention_output  # Maintain tuple output for consistency


class ComplexFeedForward(nn.Module):
    """
    A complex-valued feed-forward network.
    """

    def __init__(self, embed_dim, ff_dim):
        super(ComplexFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim, dtype=torch.cfloat)
        self.activation = ComplexGeLU()
        self.fc2 = nn.Linear(ff_dim, embed_dim, dtype=torch.cfloat)
        self.dropout = S.ComplexDropout(p=0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class ComplexTransformerEncoderBlock(nn.Module):
    """
    A single block of the Complex Transformer Encoder.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, p):
        super(ComplexTransformerEncoderBlock, self).__init__()
        self.attention = ComplexMultiHeadAttention(embed_dim, num_heads)
        self.norm1 = S.ComplexLayerNorm(embed_dim)
        self.ffn = ComplexFeedForward(embed_dim, ff_dim)
        self.norm2 = S.ComplexLayerNorm(embed_dim)
        self.dropout = S.ComplexDropout(p=p)

    def forward(self, x, return_attention=False):
        normalized_x = self.norm1(x)
        # Attention sub-layer with residual connection and layer norm
        if return_attention:
            attn_output, attn_weights = self.attention(
                normalized_x, normalized_x, normalized_x, return_attention=True
            )
        else:
            attn_output = self.attention(
                normalized_x, normalized_x, normalized_x, return_attention=False
            )

        x = x + self.dropout(attn_output)

        ffn_output = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_output)

        if return_attention:
            return x, attn_weights
        else:
            return x
