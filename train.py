import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.notebook import trange, tqdm
from config import *
from src.utils.utils import save_checkpoint, load_checkpoint, save_tokenizer, CHECKPOINT_DIR, SAVE_EVERY
from model import *
from nltk.translate.bleu_score import corpus_bleu
from model import *

def train(model, dataloader, optimizer, criterion, device, tokenizer):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        labels = input_ids.clone()
        # The criterion is initialized with ignore_index=tokenizer.pad_token_id (0).
        # So, padding tokens in 'labels' should remain tokenizer.pad_token_id to be ignored.

        optimizer.zero_grad()
        logits = model(images, input_ids, attention_mask)

        loss = criterion(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            labels[:, 1:].contiguous().view(-1)
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, tokenizer, device):
    model.eval()
    total_loss = 0
    all_references = []
    all_hypotheses = []

    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            labels = input_ids.clone()
            # The criterion is initialized with ignore_index=tokenizer.pad_token_id (0).
            # So, padding tokens in 'labels' should remain tokenizer.pad_token_id to be ignored.

            logits = model(images, input_ids, attention_mask)

            loss = criterion(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1)
            )
            total_loss += loss.item()

            for img, ref_ids in zip(images, input_ids):

                generated_caption_text = model.generate(img.unsqueeze(0), tokenizer, device=device)

                ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

                hyp_tokens = generated_caption_text.split()
                ref_tokens = ref_text.split()

                all_references.append([ref_tokens])
                all_hypotheses.append(hyp_tokens)

            if i == 0:
                sample = images[0:1]
                caption = model.generate(sample.to(device), tokenizer, device=device, temperature=1.0, top_p=0.9)
                print("\nSample caption:", caption)

        avg_loss = total_loss / len(dataloader)

        # Sửa lỗi nếu corpus_bleu nhận mảng trống
        if all_hypotheses:
            # Lưu ý: corpus_bleu cần all_references là list of list of list of tokens
            # và all_hypotheses là list of list of tokens
            bleu_4 = corpus_bleu(all_references, all_hypotheses, weights=(0.25, 0.25, 0.25, 0.25)) # BLEU-4
        else:
            bleu_4 = 0.0

    # Trả về cả loss và BLEU-4 score
    return avg_loss, bleu_4

training_loss_logger = []

list_loss_train = []
list_loss_val = []
val_bleu_score = []
best_val_loss = float("inf")
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    train_loss = train(model, train_loader, optimizer, criterion, device, tokenizer)
    list_loss_train.append(train_loss)
    print(f"Train loss: {train_loss:.4f}")
    val_loss, bleu_score = evaluate(model, val_loader, criterion, tokenizer, device)
    list_loss_val.append(val_loss)
    val_bleu_score.append(bleu_score)
    print(f"Validation loss: {val_loss:.4f}")
    print(f"Corpus BLEU-4: {bleu_score:.4f}")

    model_path = f"model_epoch_{epoch+1}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model at {model_path}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_vit_t5.pth")
        print("Saved best model")

print("Training finished.")

epochs_range = range(1, epochs + 1)
plt.figure(figsize=(15, 5))
#Train Loss vaf Validation
plt.subplot(1, 2, 1)
plt.plot(epochs_range, list_loss_train, label='Training Loss', marker = 'o')
plt.plot(epochs_range, list_loss_val, label='Validation Loss', marker = 'o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
#BLEU
plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_bleu_score, label='Validation BLEU-4', marker = 'o')
plt.title('Validation BLEU-4 Score')
plt.xlabel('Epochs')
plt.ylabel('BLEU-4 Score')
plt.legend()
plt.grid(True)
plt.show()

plt.tight_layout()
plt.show()