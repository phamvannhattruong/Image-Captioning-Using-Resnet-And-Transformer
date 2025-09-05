import json
import os
import pandas as pd
import spacy
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import Counter
from PIL import Image

class Flickr30kDataset(Dataset):
    def __init__(self, root_dir, captions_file, tokenizer, transform = None):
        self.root_dir = root_dir
        self.caption_file = captions_file
        self.df = pd.read_csv(captions_file)
        self.transform = transform
        self.tokenizer = tokenizer
        self.tokenizer.build_vocab(self.df[" comment"])
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        caption = self.df.iloc[idx, 2]
        img_id = self.df.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_id)

        # load ảnh
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # numericalize caption
        numerical_caption = [self.tokenizer.word2idx["<sos>"]]
        numerical_caption += self.tokenizer.numericalize(caption)
        numerical_caption.append(self.tokenizer.word2idx["<eos>"])

        return image, torch.tensor(numerical_caption)

class MyColatte:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)

        captions = [item(1).unsquezze(0) for item in batch]
        captions = torch.nn.utils.rnn.pad_sequence(
            captions, batch_first=True, padding_value=self.pad_idx
        )
        return images, captions

class Tokenizer:
    def __init__(self, freq_threshold=5):
        """
        freq_threshold: tần suất tối thiểu để đưa 1 từ vào vocab
        """
        self.spacy_en = spacy.load("en_core_web_sm")
        self.freq_threshold = freq_threshold

        # special tokens
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}

    def tokenize(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocab(self, sentence_list):
        counter = Counter()
        idx = 4

        for sentence in sentence_list:
            tokens = self.tokenize(sentence)
            counter.update(tokens)

        for word, freq in counter.items():
            if freq >= self.freq_threshold:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def numericalize(self, text):
        tokens = self.tokenize(text)
        return [
            self.word2idx.get(token, self.word2idx["<unk>"])
            for token in tokens
        ]

class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.spacy_en = spacy.load("en_core_web_sm")

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text):
        return [tok.text.lower() for tok in self.spacy_en.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4

        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text
        ]

def get_loader(root_dir, captions_file, freq_threshold=5, batch_size=32, num_workers=2, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Build vocab từ caption file
    with open(captions_file, "r", encoding="utf-8") as f:
        all_captions = [line.strip().split("\t")[1] for line in f]

    vocab = Vocabulary(freq_threshold)
    vocab.build_vocabulary(all_captions)

    # Dataset
    dataset = Flickr30kDataset(root_dir, captions_file, vocab, transform=transform)

    # Dataloader
    pad_idx = vocab.stoi["<PAD>"]
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset, vocab