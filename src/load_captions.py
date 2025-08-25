def load_captions(path: str):

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    captions = []
    for item in data["images"]:
        captions.extend(item["captions"])
    return captions
