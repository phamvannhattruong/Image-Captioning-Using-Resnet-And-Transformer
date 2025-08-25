def load_checkpoint(model, optimizer, path="checkpoint.pth"):

    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f" Load checkpoint từ epoch {epoch}, loss={loss}")
    return model, optimizer, epoch, loss