import torch
from model import DeepJSCC_FIS

device = "cuda" if torch.cuda.is_available() else "cpu"

model = DeepJSCC_FIS(ratio=1/12, channel_type="awgn").to(device)
model.eval()

snr = 1.0
mode = "full"

print("\n===== DEBUG FIS POWER MAP =====\n")

for b in [1.0, 0.5, 0.0]:

    print(f"\n>>> Testing budget = {b}")

    # IMPORTANT: batch dimension phải có
    x = torch.randn(1, 3, 32, 32).to(device)

    with torch.no_grad():
        z_tx, x_hat = model(x, snr=snr, budget=b, mode=mode)

    print("z_tx shape:", z_tx.shape)
    print("x_hat shape:", x_hat.shape)