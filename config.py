import os 
import torch
#dataset paths
data_set_root = "dataset/flickr8k"
images_dir = f"{data_set_root}/Images"
captions_dir = f"{data_set_root}/captions.txt"
#train set and val set paths
train_image_path = os.path.join(data_set_root, images_dir)
val_image_path = os.path.join(data_set_root, images_dir)
train_captions_file = os.path.join(data_set_root, captions_dir)
val_captions_file = os.path.join(data_set_root, captions_dir)
#hyper parameters
learning_rate = 1e-4
image_size = 128
nepochs = 200
batch_size = 16
#device 
device = "cuda" if torch.cuda.is_available() else "cpu"
#num of layers, heads, hidden size, patch size
hidden_size = 192
num_layers = (6, 6)
num_heads = 8
patch_size = 8