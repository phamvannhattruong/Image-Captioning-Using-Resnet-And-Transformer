from torch import nn
import torch
from .block import T5Block

class T5Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim=512,
        num_heads=8,
        num_layers=6,
        ff_dim=2048,
        dropout=0.1,
        max_len=64,
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.pos_embed = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList([
            T5Block(dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.fc_out = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, encoder_out, mask=None):
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_embed(input_ids) + self.pos_embed(positions)

        for layer in self.layers:
            x = layer(x, encoder_out, mask)

        x = self.norm(x)
        logits = self.fc_out(x)
        return logits