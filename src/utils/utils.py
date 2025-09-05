import json
import re
import random
from collections import Counter
from typing import List, Dict, Tuple


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", re.UNICODE)
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]

def tokenize_caption(caption: str, lowercase: bool = True) -> List[str]:

    if lowercase:
        caption = caption.lower()
    return TOKEN_RE.findall(caption)


def load_captions(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    captions = []
    for item in data["images"]:
        captions.extend(item["captions"])
    return captions

def load_captions_by_image(path: str) -> Dict[str, List[str]]:

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = {}
    for item in data["images"]:
        out[item["file_name"]] = item["captions"]
    return out

def build_vocab_from_json(path: str, min_freq: int = 1, lowercase: bool = True) -> Tuple[Dict[str, int], List[str]]:

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    captions = []
    for item in data["images"]:
        captions.extend(item["captions"])

    tokenized = [tokenize_caption(c, lowercase=lowercase) for c in captions]

    counter = Counter()
    for toks in tokenized:
        counter.update(toks)

    itos = SPECIALS.copy()
    for tok, freq in counter.most_common():
        if freq >= min_freq and tok not in SPECIALS:
            itos.append(tok)

    vocab = {tok: idx for idx, tok in enumerate(itos)}
    return vocab, itos


def tokens_to_ids(tokens: List[str], vocab: Dict[str, int], add_bos: bool = True, add_eos: bool = True) -> List[int]:

    ids = []
    if add_bos and "<bos>" in vocab:
        ids.append(vocab["<bos>"])
    for tok in tokens:
        ids.append(vocab.get(tok, vocab.get("<unk>", 1)))
    if add_eos and "<eos>" in vocab:
        ids.append(vocab["<eos>"])
    return ids

def ids_to_caption(ids: List[int], itos: List[str], skip_specials: bool = True) -> str:

    specials = set(SPECIALS)
    toks = []
    for i in ids:
        if 0 <= i < len(itos):
            t = itos[i]
            if skip_specials and t in specials:
                continue
            toks.append(t)

    s = ""
    for t in toks:
        if re.match(r"[^\w\s]", t):
            s += t
        elif s == "":
            s = t
        else:
            s += " " + t
    return s

def split_dataset(json_path: str,
                  train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  test_ratio: float = 0.1,
                  seed: int = 42) -> Dict[str, list]:

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "Tổng tỉ lệ phải = 1.0"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train_set = images[:n_train]
    val_set = images[n_train:n_train + n_val]
    test_set = images[n_train + n_val:]

    return {"train": train_set, "val": val_set, "test": test_set}


def save_checkpoint(model, optimizer, epoch: int, loss: float, path: str = "checkpoint.pth") -> None:

    import torch  
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, path)
    print(f"Đã lưu checkpoint tại: {path}")

def load_checkpoint(model, optimizer, path: str = "checkpoint.pth"):

    import torch
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Load checkpoint từ epoch {epoch}, loss={loss}")
    return model, optimizer, epoch, loss

