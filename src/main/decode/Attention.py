from torch import nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1, is_cross_attention=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert dim % num_heads == 0, "dim phải chia hết cho num_heads"
        self.is_cross_attention = is_cross_attention

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out=None, mask=None):
        B, N, D = x.shape
        q = self.q(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        if self.is_cross_attention and encoder_out is not None:
            k = self.k(encoder_out).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v(encoder_out).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        else:
            k = self.k(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            v = self.v(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, N, D)
        return self.out(out)