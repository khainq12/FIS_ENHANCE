import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from model import DeepJSCC_FIS
from explain import run_explain, save_single, save_figure3
import os
import argparse


# ================== ARGPARSE ==================
parser = argparse.ArgumentParser()
parser.add_argument("--channel", type=str, required=True)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--save_dir", type=str, default="figures")
parser.add_argument("--rayleigh_equalize", action="store_true")  # 🔥 FIX
args = parser.parse_args()


# ================== SETUP ==================
os.makedirs(args.save_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ================== DATASET ==================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# ================== MODEL ==================
model = DeepJSCC_FIS().to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device))
model.eval()

print(f"✅ Loaded checkpoint: {args.ckpt}")
print(f"Channel: {args.channel}")
print(f"Rayleigh Equalization: {args.rayleigh_equalize}")


# ================== SELECT IMAGES ==================
img1 = dataset[10][0]
img2 = dataset[50][0]
img3 = dataset[100][0]


# ================== RUN EXPLAIN ==================
res1 = run_explain(
    model, img1, 1, args.channel, device,
    rayleigh_equalize=args.rayleigh_equalize
)

res2 = run_explain(
    model, img2, 7, args.channel, device,
    rayleigh_equalize=args.rayleigh_equalize
)

res3 = run_explain(
    model, img3, 13, args.channel, device,
    rayleigh_equalize=args.rayleigh_equalize
)


# ================== SAVE SINGLE ==================
save_single(*res1, f"{args.channel} SNR=1", f"{args.save_dir}/snr1.png")
save_single(*res2, f"{args.channel} SNR=7", f"{args.save_dir}/snr7.png")
save_single(*res3, f"{args.channel} SNR=13", f"{args.save_dir}/snr13.png")


# ================== SAVE FIGURE 3 ==================
save_figure3(
    [res1, res2, res3],
    [
        f"{args.channel} (low SNR)",
        f"{args.channel} (mid SNR)",
        f"{args.channel} (high SNR)"
    ],
    f"{args.save_dir}/figure3.png"
)

print(f"✅ Saved results to {args.save_dir}")