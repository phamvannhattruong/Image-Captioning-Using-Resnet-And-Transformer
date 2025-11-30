import os
from torch.utils.data import DataLoader
from torchvision import transforms
from src.main.model import ViT_Transformer
from src.data.dataset import JsonCaptionsDataset
import torch
import torch.optim as optim
import torch.nn as nn
from config import *
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from transformers import T5Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = T5Tokenizer.from_pretrained("t5-small")

if tokenizer.bos_token_id is None:
    tokenizer.bos_token_id = tokenizer.pad_token_id

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_data = JsonCaptionsDataset(
    train_image_path, train_caption_path,
    image_transform=transform,
    caption_tokenizer=tokenizer,
    max_len=trans_cfg["max_len"]
)
val_data = JsonCaptionsDataset(
    val_image_path, val_caption_path,
    image_transform=transform,
    caption_tokenizer=tokenizer,
    max_len=trans_cfg["max_len"]
)

# train_data = torch.utils.data.Subset(train_data, range(500))
# val_data = torch.utils.data.Subset(val_data, range(100))

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

print(f"ViT embed_dim: {vit_cfg['embed_dim']}, T5 dim: {trans_cfg['dim']}")
print(f"Tokenizer size: {len(tokenizer)}")
# Tính toán và kiểm tra num_patches:
num_patches = (vit_cfg['image_size'] // vit_cfg['patch_size']) ** 2
print(f"ViT Patch Count + 1: {num_patches + 1}")
print(f"Transformer Max Length: {trans_cfg['max_len']}")

model = ViT_Transformer(vit_cfg, trans_cfg, len(tokenizer)).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)

criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

print(f"Mô hình ViT_trans_cfg đã được khởi tạo trên: {device}")
print(f"Kích thước từ vựng: {len(tokenizer)}")
print(f"Loss Function sẽ bỏ qua token ID: {tokenizer.pad_token_id} (PAD)")