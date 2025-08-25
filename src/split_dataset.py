def split_dataset(json_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data["images"]
    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    n_test  = n - n_train - n_val

    train_set = images[:n_train]
    val_set   = images[n_train:n_train+n_val]
    test_set  = images[n_train+n_val:]

    return {
        "train": train_set,
        "val": val_set,
        "test": test_set
    }
