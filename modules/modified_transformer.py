from typing import Optional
from torch.nn import functional as F
import torch
from einops import rearrange
from torch import nn, Tensor
from torch.nn import init, MultiheadAttention, Linear, Dropout, BatchNorm1d


class TransformerBatchNormEncoderLayer(nn.modules.Module):
    r"""This transformer encoder layer block is made up of self-attn and feedforward network.
    It differs from TransformerEncoderLayer in torch/nn/modules/transformer.py in that it replaces LayerNorm
    with BatchNorm.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, pre_norm=False):
        super(TransformerBatchNormEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = BatchNorm1d(d_model, eps=1e-5)  # normalizes each feature across batch samples and time steps
        self.norm2 = BatchNorm1d(d_model, eps=1e-5)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.activation = F.gelu

        self.pre_norm = pre_norm

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerBatchNormEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None, **kwargs) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # src of shape [batch_size, sequence_length, feature_dim]
        if not self.pre_norm:
            src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2)  # (batch_size, seq_len, d_model)
            src = src.permute(0, 2, 1)  # (batch_size, d_model, seq_len)
            src = self.norm1(src)
            src = src.permute(0, 2, 1)  # restore (batch_size, seq_len, d_model)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
            src = src + self.dropout2(src2)  # (batch_size, seq_len, d_model)
            src = src.permute(0, 2, 1)  # (batch_size, d_model, seq_len)
            src = self.norm2(src)
            src = src.permute(0, 2, 1)  # restore (batch_size, seq_len, d_model)
        else: # post-norm
            # src of shape [batch_size, sequence_length, feature_dim]
            src1 = self.norm1(src.permute(0, 2, 1)) # (batch_size, d_model, seq_len)
            src1 = src1.permute(0, 2, 1) # restore (batch_size, seq_len, d_model)
            src2 = self.self_attn(src1, src1, src1, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]
            src = src + self.dropout1(src2) # (batch_size, seq_len, d_model)
            src1 = self.norm2(src.permute(0, 2, 1))  # (batch_size, d_model, seq_len)
            src1 = src1.permute(0, 2, 1) # restore (batch_size, seq_len, d_model)
            src2 = self.linear2(self.dropout(self.activation(self.linear1(src1))))
            src = src + self.dropout2(src2) # (batch_size, seq_len, d_model)
        return src


class KDDOriginalLearnablePositionalEncoding(nn.Module):
    def __init__(self, length, dim):
        super(KDDOriginalLearnablePositionalEncoding, self).__init__()
        self.length = length
        self.positional_embedding = nn.Parameter(torch.empty(1, self.length, dim))
        nn.init.uniform_(self.positional_embedding, -0.02, 0.02)

    def forward(self, x):
        # print(f"The shape of x is: {x.shape}")
        pos_embeddings = self.positional_embedding[:, :x.size(1), :]
        x = x + pos_embeddings
        return x


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, length, dim):
        super(LearnablePositionalEncoding, self).__init__()
        self.length = length
        self.positional_embedding = nn.Embedding(self.length, dim)
        init.normal_(self.positional_embedding.weight, std=0.02)  # same

    def forward(self, x):
        positions = torch.arange(self.length, dtype=torch.long, device=x.device)
        all_positions = positions.unsqueeze(0).expand(x.shape[0], -1)
        pos_embeddings = self.positional_embedding(all_positions)
        pos_embeddings = pos_embeddings.view(x.shape)
        x = x + pos_embeddings
        return x


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
