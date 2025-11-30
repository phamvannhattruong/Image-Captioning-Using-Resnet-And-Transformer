import os, json
import random 
from torch import nn
from PIL import Image
from torch.utils.data import Dataset

class JsonCaptionsDataset(Dataset):
    def __init__(self, root, annFile, transform=None, target_transform=None,
                 img_key="file_name", cap_key="comment"):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.img_key = img_key
        self.cap_key = cap_key

        with open(annFile, "r", encoding="utf-8") as f:
            data = json.load(f)
        records = data["images"] if isinstance(data, dict) and "images" in data else data

        self.items = []
        for d in records:
            fn = d[self.img_key]
            caps = d.get(self.cap_key, [])
            if not caps:
                continue
            self.items.append((os.path.join(root, fn), caps))

        if not self.items:
            raise RuntimeError("Không tìm thấy mẫu nào. Kiểm tra đường dẫn ảnh và khóa 'file_name'/'comment' trong captions.json.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, caps = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = caps
        if target_transform := self.target_transform:
            target = target_transform(target)  # SampleCaption sẽ chọn 1 câu từ list
        return img, target
    
class SampleCaption(nn.Module):
    def __call__(self, sample):
        # sample ở đây là list caption (trong JSON key "comment")
        rand_index = random.randint(0, len(sample) - 1)
        return sample[rand_index]
