# git status
# git add .
# git commit -m "Fix training bug + improve FIS"
# git push



# # 1. Xem file nào đã thay đổi nhưng chưa commit
# git status

# # 2. Thêm file mới vào git
# git add .

# # 3. Commit
# git commit -m "Add run_all.sh, fix diagnose_controller, add paper sims"
#GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519" git push -u origin feature/fis-channel-aware-controller-r2

# # 4. Push
# git push -u origin feature/fis-channel-aware-controller-r2
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