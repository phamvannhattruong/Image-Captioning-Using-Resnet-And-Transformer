import json
import os
from torch.utils.data import Dataset
from PIL import Image

class ImageCaptionDataset(Dataset):
    def __init__(self, data_dir, captions_file, image_transform = None, caption_transform = None):
        self.data_dir = data_dir
        self.image_transform = image_transform
        self.caption_transform = caption_transform

        with open(captions_file, "r", encoding='utf-8') as f:
            self.caption_data = json.loads(f)

        self.samples = []

        for img_id, caption in self.caption_data.items():
            for cap in caption:
                self.samples.append(caption)
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        img_id, caption = self.samples[idx]

        img_path = os.path.join(self.data_dir, img_id)
        image = Image.open(img_path).convert("RGB")

        if self.image_transform:
            image = self.image_transform(image)

        if self.caption_transform:
            caption = self.caption_transform(caption)

        return image, caption, img_id



