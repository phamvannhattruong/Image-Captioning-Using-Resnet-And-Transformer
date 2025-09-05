import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from src.data.dataset import Flickr30kDataset
from src.llm.model import transformer
from src.resnet.ResNet import ResNet50

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"










