import torch

from torch import nn

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim = -1) #use softmax to calculate last dimension
    def forward(self, q, k, v, mask=None, e = 1e-12):
        batch_size, head, length, d_tensor = k.size() # k.size() -> [number of batch, number of attention head, length of token, size of embedding]
        k_transpose = torch.transpose(2, 3)
        score = q @ k_transpose/ torch.sqrt(d_tensor)
        if mask is None:
            score = score.masked_fill(mask==0, -100000)
        score = self.softmax(score)
        attention = score * v
        return v, attention


