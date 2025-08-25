def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint.pth"):

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, path)
    print(f"Đã lưu checkpoint tại {path}")