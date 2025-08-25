import numpy as np
from torch import nn
from src.llm.embedding.tranformer_embedding import TransformerEmbedding
from src.llm.blocks.decoder_layers import DecoderLayer

class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(dimension = d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)
        self.layers = nn.ModuleList([DecoderLayer(d_model = d_model,
                                                  ffn_hidden = ffn_hidden,
                                                       n_head = n_head,
                                                  drop_prob = drop_prob)])
        self.linear = nn.Linear(d_model, dec_voc_size)
    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        output = self.linear(trg)
        return output