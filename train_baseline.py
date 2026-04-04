
import argparse
import json
import os
import random
import time
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import create_dataset
from model_baseline import DeepJSCC, ratio2filtersize
from utils import get_psnr


def parse_snr_list(values: List[float]) -> List[float]:
    vals = [float(v) for v in values]
    if len(vals) == 0:
        raise ValueError("eval_snr_list must not be empty.")
    return vals


@torch.no_grad()
def evaluate_multi_snr(model, loader, device, snr_list, channel, rician_k):
    model.eval()
    total_psnr = 0.0
    for snr in snr_list:
        model.change_channel(channel_type=channel, snr=snr, rician_k=rician_k)
        psnr_sum, n = 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            x_hat = model(x)
            psnr = get_psnr(x_hat * 255.0, x * 255.0, max_val=255.0).mean().item()
            psnr_sum += psnr
            n += 1
        total_psnr += psnr_sum / max(n, 1)
    return total_psnr / len(snr_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=1 / 6)
    ap.add_argument("--channel", type=str, default="AWGN",
                    choices=["AWGN", "Rayleigh", "Rician", "rayleigh_legacy"])
    ap.add_argument("--rician_k", type=float, default=4.0)
    ap.add_argument("--rayleigh_equalize", action="store_true",
                    help="Enable Rayleigh equalization (same as FIS)")
    ap.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "celebahq", "folder"])
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--image_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--snr_min", type=float, default=0.0)
    ap.add_argument("--snr_max", type=float, default=20.0)
    ap.add_argument("--eval_snr_list", type=float, nargs="+", default=None,
                    help="Use the same SNR list as FIS for fair best-checkpoint selection.")
    ap.add_argument("--save_dir", type=str, default="ckpts_baseline")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--random_flip", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = create_dataset(
        args.dataset, split="train", data_root=args.data_root,
        image_size=args.image_size, random_flip=args.random_flip
    )
    testset = create_dataset(
        args.dataset, split="test", data_root=args.data_root,
        image_size=args.image_size, random_flip=False
    )

    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True
    )
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )

    image_first = trainset[0][0]
    c = ratio2filtersize(image_first, args.ratio)

    eval_snr_list = (
        parse_snr_list(args.eval_snr_list)
        if args.eval_snr_list is not None
        else [0.5 * (args.snr_min + args.snr_max)]
    )

    meta = {
        "channel": args.channel,
        "rician_k": args.rician_k,
        "dataset": args.dataset,
        "image_size": args.image_size,
        "snr_min": args.snr_min,
        "snr_max": args.snr_max,
        "eval_snr_list": eval_snr_list,
        "rayleigh_equalize": bool(args.rayleigh_equalize),
        "seed": args.seed,
        "ratio": args.ratio,
    }
    with open(os.path.join(args.save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    model = DeepJSCC(
        c=c,
        channel_type=args.channel,
        snr=random.uniform(args.snr_min, args.snr_max),
        rician_k=args.rician_k,
    ).to(device)

    if hasattr(model, "channel") and hasattr(model.channel, "enable_rayleigh_equalization"):
        model.channel.enable_rayleigh_equalization(args.rayleigh_equalize)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    best_psnr = -1.0

    print(f"Eval SNR list for checkpoint selection: {eval_snr_list}")

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0

        for x, _ in train_loader:
            x = x.to(device)
            snr = args.snr_min if args.snr_max <= args.snr_min else random.uniform(args.snr_min, args.snr_max)
            model.change_channel(channel_type=args.channel, snr=snr, rician_k=args.rician_k)
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        psnr_avg = evaluate_multi_snr(
            model, test_loader, device, eval_snr_list, args.channel, args.rician_k
        )

        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            torch.save(model.state_dict(), os.path.join(args.save_dir, "baseline_best.pth"))

        torch.save(model.state_dict(), os.path.join(args.save_dir, f"baseline_ep{ep:03d}.pth"))
        print(
            f"Epoch {ep:03d} | loss={running / len(train_loader):.6f} "
            f"| mean_PSNR={psnr_avg:.3f} | time={time.time() - t0:.1f}s"
        )

    print("Best mean PSNR:", best_psnr)


if __name__ == "__main__":
    main()
