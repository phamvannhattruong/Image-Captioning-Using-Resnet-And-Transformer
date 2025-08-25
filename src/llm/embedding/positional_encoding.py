import torch
from torch import nn

class PositionEncoding(nn.Module):
    def __init__(self, dimension, max_len, device):
        super(PositionEncoding, self).__init__()

        self.encoding = torch.zeros(max_len, dimension, device=device) #create matrix with height is dimension and width is max_len
        self.encoding.requires_grad = False #No need to compute gradient

        pos = torch.arange(0, max_len, device = device)
        pos = pos.float().unsqueeze(dim = 1) #turn into column vector

        #create even index
        _2i = torch.arange(0, max_len, step=2, device = device)

        #calculate positional encoding
        self.encoding[:, 0::2] = torch.sin(pos/(10000**(_2i/dimension))) #even index dimension
        self.encoding[:, 1::2] = torch.cos(pos/(10000**(_2i/dimension))) #odd index dimension
    def forward(self, x):
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]


