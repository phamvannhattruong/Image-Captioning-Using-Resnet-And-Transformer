def tokens_to_ids(tokens, vocab, add_bos=True, add_eos=True):
    ids = []
    if add_bos and "<bos>" in vocab:
        ids.append(vocab["<bos>"])
    
    for tok in tokens:
        if tok in vocab:
            ids.append(vocab[tok])
        else:
            ids.append(vocab.get("<unk>", 1)) 
    
    if add_eos and "<eos>" in vocab:
        ids.append(vocab["<eos>"])
    
    return ids