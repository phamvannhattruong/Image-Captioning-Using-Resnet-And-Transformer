from torch import nn
from .attention import MultiHeadAttention
from .feedforward import FeedForward

class T5Block(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(dim)
        self.cross_attn_norm = nn.LayerNorm(dim)
        self.ff_norm = nn.LayerNorm(dim)

        self.self_attn = MultiHeadAttention(dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(dim, num_heads, dropout, is_cross_attention=True)
        self.ff = FeedForward(dim, ff_dim, dropout)

    def forward(self, x, encoder_out, mask=None):
        # Self-attention (decoder attends to previous tokens)
        x = x + self.self_attn(self.self_attn_norm(x), mask=mask)
        # Cross-attention (decoder attends to encoder outputs)
        x = x + self.cross_attn(self.cross_attn_norm(x), encoder_out)
        # Feed-forward
        x = x + self.ff(self.ff_norm(x))
        return x