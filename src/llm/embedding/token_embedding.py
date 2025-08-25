import torch
from torch import nn

class ToKenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, dimension):
        super (ToKenEmbedding, self).__init__(vocab_size, dimension, padding_idx=1)