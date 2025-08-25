import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, dimension, eps = 1e-12):
        self.gamma = nn.Parameter(torch.ones(dimension))
        self.beta = nn.Parameter(torch.zeros(dimension))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True) #calculate average the last dimension of tensor
        var = x.var(-1, unbiased=False, keepdim=True) #calcutlate varience

        out = (x - mean)/torch.sqrt(var + self.eps)
        out = self.gamma*out + self.beta

        return out

