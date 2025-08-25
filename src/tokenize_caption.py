def tokenize_caption(caption: str, lowercase: bool = True):
    TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|[^\w\s]", re.UNICODE)

    if lowercase:
        caption = caption.lower()
    tokens = TOKEN_RE.findall(caption)
    return tokens