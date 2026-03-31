# git status
# git add .
# git commit -m "Fix training bug + improve FIS"
# git push
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
from model import DeepJSCC_FIS   # sửa đúng path của bạn

device = "cuda" if torch.cuda.is_available() else "cpu"

# dataset
transform = transforms.ToTensor()
dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)

# model
model = DeepJSCC_FIS().to(device)
model.eval()

# lấy 1 ảnh
img = dataset[0][0]
x = img.unsqueeze(0).to(device)

# chạy
z_tx, x_hat, info = model(x, snr=7, return_info=True)

print(info.keys())