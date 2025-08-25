def ids_to_caption(ids, itos, skip_specials=True):
    specials = {"<pad>", "<bos>", "<eos>", "<unk>"}
    tokens = []

    for i in ids:
        if i < 0 or i >= len(itos):
            continue
        tok = itos[i]
        if skip_specials and tok in specials:
            continue
        tokens.append(tok)

    caption = ""
    for t in tokens:
        if re.match(r"[^\w\s]", t):  
            caption += t
        elif caption == "":
            caption = t
        else:
            caption += " " + t
    return caption
