import torch
from torch import nn
from src.resnet.ResNet import ResNet50
from src.llm.model.transformer import Transformer
class ImageCaptioningModel(nn.Module):
    def __init__(self, emb_size, vocab_size, transformer_config, device):
        super(ImageCaptioningModel, self).__init__()
        self.device = device
        self.resnet = ResNet50(num_classes=emb_size)
        self.transformer = Transformer(**transformer_config)
        self.img_fc = nn.Linear(emb_size, transformer_config["d_model"])
    def forward(self, images, captions):
        feature = self.resnet(images)
        feature = self.img_fc(feature)

        features = feature.unsqueeze(1)

        output = self.transformer(features, captions)
        return output

