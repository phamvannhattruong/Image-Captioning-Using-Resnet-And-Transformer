def build_vocab_from_json(path, min_freq=1, lowercase=True):
    TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", re.UNICODE)

    def tokenize(text):
        if lowercase:
            text = text.lower()
        return TOKEN_RE.findall(text)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    captions = []
    for item in data["images"]:
        captions.extend(item["captions"])

    tokenized = [tokenize(c) for c in captions]

    counter = Counter()
    for toks in tokenized:
        counter.update(toks)

    specials = ["<pad>", "<unk>", "<bos>", "<eos>"]
    itos = specials.copy()
    for tok, freq in counter.most_common():
        if freq >= min_freq and tok not in specials:
            itos.append(tok)

    vocab = {tok: idx for idx, tok in enumerate(itos)}

    return vocab, itos