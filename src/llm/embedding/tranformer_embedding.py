from torch import nn
from token_embedding import ToKenEmbedding
from positional_encoding import PositionEncoding

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, dimension, max_len, drop_prob, device ):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = ToKenEmbedding(vocab_size, dimension) #token ID -> vector embedding
        self.pos_emb = PositionEncoding(dimension, max_len, device) #create matrix positional encoding
        self.drop_out = nn.Dropout(p=drop_prob) #drop out to avoid overfitting
    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)