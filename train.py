import torch
import torch.optim as optim
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.notebook import trange, tqdm
from config import *
from src.data.dataset import JsonCaptionsDataset, SampleCaption
from src.main.model.ImageCaptioningModel import VisionEncoderDecoder
from src.token.TokenDrop import TokenDrop
from src.utils.utils import save_checkpoint, load_checkpoint, save_tokenizer, CHECKPOINT_DIR, SAVE_EVERY

train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.5)
])

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

train_dataset = JsonCaptionsDataset(
    root=train_image_path,
    annFile=train_captions_file,
    transform=train_transform,
    target_transform=SampleCaption()
)

val_dataset = JsonCaptionsDataset(
    root=val_image_path,
    annFile=val_captions_file,
    transform=transform,
    target_transform=SampleCaption()
)

data_loader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
data_loader_val   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
dataiter = next(iter(data_loader_val))
test_images, test_captions = dataiter

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

caption_model = VisionEncoderDecoder(
    image_size=image_size,
    channels_in=test_images.shape[1],
    num_emb=tokenizer.vocab_size,
    patch_size=patch_size,
    num_layers=num_layers,
    hidden_size=hidden_size,
    num_heads=num_heads
).to(device)

optimizer = optim.Adam(caption_model.parameters(), lr=learning_rate)
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=tokenizer.pad_token_id)
td = TokenDrop(prob=0.5, blank_token=tokenizer.pad_token_id, eos_token=tokenizer.sep_token_id)

training_loss_logger = []

for epoch in trange(0, nepochs, leave=False, desc="Epoch"):
    # Set the model in training mode
    caption_model.train()
    steps = 0
    # Iterate over the training data loader
    for images, captions in tqdm(data_loader_train, desc="Training", leave=False):

        images = images.to(device)

        # Tokenize and pre-process the captions
        tokens = tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
        token_ids = tokens['input_ids'].to(device)
        padding_mask = tokens['attention_mask'].to(device)
        bs = token_ids.shape[0]

        # Shift the input sequence to create the target sequence
        target_ids = torch.cat((token_ids[:, 1:],
                                torch.zeros(bs, 1, device=device).long()), 1)

        tokens_in = td(token_ids)
        with torch.cuda.amp.autocast():
            # Forward pass
            pred = caption_model(images, tokens_in, padding_mask=padding_mask)

        # Compute the loss
        loss = (loss_fn(pred.transpose(1, 2), target_ids) * padding_mask).mean()

        # Backpropagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Log the training loss
        training_loss_logger.append(loss.item())

        # (tuỳ chọn) nếu muốn đếm step:
        steps += 1

    # ============================
    # LƯU CHECKPOINT SAU MỖI EPOCH
    # ============================
    # Chỉnh: chỉ lưu mỗi SAVE_EVERY epoch
    if (epoch + 1) % SAVE_EVERY == 0:
        last_ckpt_path = os.path.join(CHECKPOINT_DIR, "last.pt")
        save_checkpoint(caption_model, optimizer, scaler, epoch, steps, last_ckpt_path)
        save_tokenizer(tokenizer, CHECKPOINT_DIR)
        print(f"[SAVE] Epoch {epoch} -> {last_ckpt_path}")