
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
from model import DeepJSCC_FIS
from model_baseline import ratio2filtersize
from utils import get_psnr


def parse_snr_list(values: List[float]) -> List[float]:
    vals = [float(v) for v in values]
    if len(vals) == 0:
        raise ValueError("train/eval SNR list must not be empty.")
    return vals


def compute_rule_usage(model, loader, device, snr, budget, mode):
    model.eval()
    rule1_accum = None
    rule2_accum = None
    count = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            _, _, info = model(x, snr=snr, budget=budget, mode=mode, return_info=True)
            if "rule1_strength" in info:
                r1 = info["rule1_strength"].mean(dim=(0, 2, 3))
                rule1_accum = r1 if rule1_accum is None else rule1_accum + r1
            if "rule2_strength" in info:
                r2 = info["rule2_strength"].mean(dim=(0, 2, 3))
                rule2_accum = r2 if rule2_accum is None else rule2_accum + r2
            count += 1

    result = {}
    if rule1_accum is not None and count > 0:
        r1_final = rule1_accum / count
        r1_final = r1_final / r1_final.sum()
        result["layer1"] = r1_final.cpu().tolist()
    if rule2_accum is not None and count > 0:
        r2_final = rule2_accum / count
        r2_final = r2_final / r2_final.sum()
        result["layer2"] = r2_final.cpu().tolist()
    return result


@torch.no_grad()
def evaluate_multi_snr(model, loader, device, snr_list, budget, mode, channel, rician_k):
    model.eval()
    total_psnr = 0.0
    for snr in snr_list:
        model.set_channel(channel_type=channel, snr=snr, rician_k=rician_k)
        psnr_sum, n = 0.0, 0
        for x, _ in loader:
            x = x.to(device)
            _, x_hat = model(x, snr=snr, budget=budget, mode=mode, return_info=False)
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
    ap.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "celebahq", "folder"])
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--image_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--snr_min", type=float, default=0.0)
    ap.add_argument("--snr_max", type=float, default=20.0)
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--mode", type=str, default="full",
                    choices=["full", "linear", "importance_only", "snr_only"])
    ap.add_argument("--save_dir", type=str, default="ckpts_fis_power")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--random_flip", action="store_true")
    ap.add_argument("--rayleigh_equalize", action="store_true")
    ap.add_argument(
        "--train_snr_list",
        type=float,
        nargs="+",
        default=None,
        help="Explicit SNR grid used during training, e.g. --train_snr_list 1 4 7 10 13",
    )
    ap.add_argument(
        "--eval_snr_list",
        type=float,
        nargs="+",
        default=None,
        help="SNRs used for best-checkpoint selection. Defaults to train_snr_list; if that is also omitted, uses midpoint range.",
    )
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = create_dataset(
        args.dataset,
        split="train",
        data_root=args.data_root,
        image_size=args.image_size,
        random_flip=args.random_flip,
    )
    testset = create_dataset(
        args.dataset,
        split="test",
        data_root=args.data_root,
        image_size=args.image_size,
        random_flip=False,
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

    train_snr_list = (
        parse_snr_list(args.train_snr_list)
        if args.train_snr_list is not None
        else [args.snr_min, 0.5 * (args.snr_min + args.snr_max), args.snr_max]
    )
    eval_snr_list = (
        parse_snr_list(args.eval_snr_list)
        if args.eval_snr_list is not None
        else train_snr_list
    )

    meta = {
        "channel": args.channel,
        "rician_k": args.rician_k,
        "dataset": args.dataset,
        "image_size": args.image_size,
        "budget": args.budget,
        "mode": args.mode,
        "snr_min": args.snr_min,
        "snr_max": args.snr_max,
        "train_snr_list": train_snr_list,
        "eval_snr_list": eval_snr_list,
        "rayleigh_equalize": bool(args.rayleigh_equalize),
        "seed": args.seed,
        "ratio": args.ratio,
    }
    with open(os.path.join(args.save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    model = DeepJSCC_FIS(
        ratio=args.ratio,
        c=c,
        P=1.0,
        channel_type=args.channel,
        rician_k=args.rician_k,
        snr_max_db=args.snr_max,
        snr_min_db=args.snr_min,
    ).to(device)
    model.channel.enable_rayleigh_equalization(args.rayleigh_equalize)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_psnr = -1.0

    print(f"Training SNR list: {train_snr_list}")
    print(f"Eval SNR list: {eval_snr_list}")

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0

        for x, _ in train_loader:
            x = x.to(device)
            snr = random.choice(train_snr_list)
            model.set_channel(channel_type=args.channel, snr=snr, rician_k=args.rician_k)
            _, x_hat = model(x, snr=snr, budget=args.budget, mode=args.mode, return_info=False)
            loss = loss_fn(x_hat, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        psnr_avg = evaluate_multi_snr(
            model,
            test_loader,
            device,
            snr_list=eval_snr_list,
            budget=args.budget,
            mode=args.mode,
            channel=args.channel,
            rician_k=args.rician_k,
        )

        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            torch.save(model.state_dict(), os.path.join(args.save_dir, "fis_power_best.pth"))
            print(
                f">>> New BEST model (mean PSNR over {eval_snr_list} dB = {psnr_avg:.3f}) "
                f"-> computing rule usage..."
            )
            snr_diag = float(sum(eval_snr_list) / len(eval_snr_list))
            rule_usage = compute_rule_usage(
                model, test_loader, device, snr=snr_diag, budget=args.budget, mode=args.mode
            )
            with open(os.path.join(args.save_dir, "rule_usage_best.json"), "w", encoding="utf-8") as f:
                json.dump(rule_usage, f, indent=2)

        torch.save(model.state_dict(), os.path.join(args.save_dir, f"fis_power_ep{ep:03d}.pth"))
        print(
            f"Epoch {ep:03d} | loss={running / len(train_loader):.6f} "
            f"| mean_PSNR={psnr_avg:.3f} dB | time={time.time() - t0:.1f}s"
        )

    print("Best mean PSNR:", best_psnr)


if __name__ == "__main__":
    main()
