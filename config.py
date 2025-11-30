#dataset path
data_root = "/content/drive/MyDrive/Image_captioning/flickr8k"
image_dir = f"{data_root}/Images"
caption_dir = f"{data_root}/captions/caption_8k.json"
#train set and val set
train_image_path = image_dir
train_caption_path = caption_dir
val_image_path = image_dir
val_caption_path = caption_dir
#ViT config and transformer config
# Using smaller, diagnostic configuration to troubleshoot CUDA error
vit_cfg = dict(
    image_size=224,
    patch_size=16,
    in_channels=3,
    embed_dim=128, # Reduced dimension
    depth=2,       # Reduced depth
    num_heads=4,   # Reduced heads
    mlp_ratio=2.0,
    dropout=0.1
)

trans_cfg = dict(
    dim=128,       # Reduced dimension
    num_heads=4,   # Reduced heads
    num_layers=2,  # Reduced layers
    ff_dim=256,    # Reduced feed-forward dimension
    dropout=0.1,
    max_len=10     # Reduced max_len
)

epochs = 200