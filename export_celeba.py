from datasets import load_dataset
import os
from tqdm import tqdm

ds = load_dataset("korexyz/celeba-hq-256x256")

out_dir = "/media/data/students/nguyenquangkhai/celeba256"
os.makedirs(out_dir, exist_ok=True)

for i, sample in enumerate(tqdm(ds["train"])):
    img = sample["image"]
    img.save(f"{out_dir}/{i:06d}.png")