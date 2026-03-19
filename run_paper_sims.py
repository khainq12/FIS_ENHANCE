import argparse
import json
import os
import time
from collections import Counter

import torch
from torch.utils.data import DataLoader

from dataset import create_dataset
from channel import Channel
from model_baseline import DeepJSCC as DeepJSCC_Baseline, ratio2filtersize
from model import DeepJSCC_FIS, power_normalize
from utils import get_psnr, simple_ssim


def parse_list(s: str, cast=float):
    if s is None or s == "":
        return []
    return [cast(x.strip()) for x in s.split(",") if x.strip() != ""]


@torch.no_grad()
def eval_one(
    baseline_model,
    fis_models,  # dict {mode: model}
    channel_type: str,
    snr_db: float,
    budget: float,
    modes,
    loader,
    device,
    seed_base: int = 1234,
    rayleigh_equalize: bool = False,
    max_batches: int = None,
    collect_explain: bool = False,
    rician_k: float = 4.0,
):
    channel = Channel(channel_type=channel_type, snr_db=snr_db, rician_k=rician_k)
    channel.enable_rayleigh_equalization(rayleigh_equalize)
    channel.change_snr(snr_db)

    results = {m: {"psnr": 0.0, "ssim": 0.0, "n": 0} for m in modes}
    rule_counts = {m: {"layer1": Counter(), "layer2": Counter()} for m in modes}

    for bidx, (x, _) in enumerate(loader):
        if max_batches is not None and bidx >= max_batches:
            break

        x = x.to(device)

        # 🔥 SAME NOISE FOR ALL METHODS
        torch.manual_seed(seed_base + bidx)

        # ================= BASELINE =================
        t0 = time.time()

        z = baseline_model.encoder(x)
        z = power_normalize(z, P=1.0, eps=1e-8)
        z_noisy = channel(z)
        x_hat = baseline_model.decoder(z_noisy)

        elapsed = time.time() - t0

        results["baseline"].setdefault("time", 0.0)
        results["baseline"]["time"] += elapsed

        psnr = get_psnr(x_hat * 255.0, x * 255.0, max_val=255.0).mean().item()
        ssim = simple_ssim(x_hat, x).mean().item()

        results["baseline"]["psnr"] += psnr
        results["baseline"]["ssim"] += ssim
        results["baseline"]["n"] += 1

        # ================= FIS MODES =================
        for mode in modes:
            if mode == "baseline":
                continue

            fis_model = fis_models[mode]

            t0 = time.time()

            z = fis_model.encoder(x)

            if collect_explain:
                A, info = fis_model.controller(
                    z, snr_db=snr_db, budget=budget,
                    mode=mode, return_info=True
                )
            else:
                A = fis_model.controller(
                    z, snr_db=snr_db, budget=budget,
                    mode=mode, return_info=False
                )
                info = None

            gain = torch.sqrt(A.clamp_min(fis_model.eps))
            z_g = z * gain.unsqueeze(1)
            z_tx = power_normalize(z_g, P=fis_model.P, eps=fis_model.eps)

            # 🔥 SAME CHANNEL REALIZATION
            z_noisy = channel(z_tx)

            x_hat = fis_model.decoder(z_noisy)

            elapsed = time.time() - t0

            results[mode].setdefault("time", 0.0)
            results[mode]["time"] += elapsed

            psnr = get_psnr(x_hat * 255.0, x * 255.0, max_val=255.0).mean().item()
            ssim = simple_ssim(x_hat, x).mean().item()

            results[mode]["psnr"] += psnr
            results[mode]["ssim"] += ssim
            results[mode]["n"] += 1

            if collect_explain and info is not None:
                if "rule1_id" in info:
                    rule_counts[mode]["layer1"].update(
                        info["rule1_id"].detach().cpu().view(-1).tolist()
                    )
                if "rule2_id" in info:
                    rule_counts[mode]["layer2"].update(
                        info["rule2_id"].detach().cpu().view(-1).tolist()
                    )

    for mode in modes:
        n = max(results[mode]["n"], 1)
        results[mode]["psnr"] /= n
        results[mode]["ssim"] /= n
        results[mode]["time"] /= n

    return results, rule_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_ckpt", type=str, required=True)
    ap.add_argument("--fis_ckpt_root", type=str, required=True)

    ap.add_argument("--ratio", type=float, default=1 / 12)
    ap.add_argument("--channel", type=str, default="AWGN",
                    choices=["AWGN", "Rayleigh", "Rician", "rayleigh_legacy"])
    ap.add_argument("--rician_k", type=float, default=4.0)

    ap.add_argument("--snrs", type=str, default="1,4,7,10,13")
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--budgets", type=str, default="")

    ap.add_argument("--modes", type=str,
                    default="baseline,linear,importance_only,snr_only,full")

    ap.add_argument("--dataset", type=str, default="cifar10")
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--image_size", type=int, default=32)

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--save_dir", type=str, default="paper_sims_out")

    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)

    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== DATA =====
    testset = create_dataset(
        args.dataset,
        split="test",
        data_root=args.data_root,
        image_size=args.image_size,
        random_flip=False,
    )
    loader = DataLoader(testset, batch_size=args.batch_size,
                        shuffle=False, num_workers=args.num_workers)

    x0, _ = testset[0]
    c = ratio2filtersize(x0, args.ratio)
    print("Latent channel size c =", c)

    # ===== BASELINE =====
    baseline_model = DeepJSCC_Baseline(
        c=c, channel_type=args.channel, rician_k=args.rician_k
    ).to(device)

    ckpt = torch.load(args.baseline_ckpt, map_location=device)
    baseline_model.load_state_dict(ckpt, strict=True)
    baseline_model.eval()

    # ===== LOAD ALL FIS =====
    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]

    fis_models = {}
    for mode in modes:
        if mode == "baseline":
            continue

        ckpt_path = os.path.join(
            args.fis_ckpt_root,
            f"ckpts_{mode}_{args.channel}",
            "fis_power_best.pth"
        )

        model = DeepJSCC_FIS(
            c=c,
            ratio=args.ratio,
            channel_type=args.channel,
            rician_k=args.rician_k,
        ).to(device)

        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt, strict=True)
        model.eval()

        fis_models[mode] = model

    snrs = parse_list(args.snrs, float)
    budgets = parse_list(args.budgets, float) if args.budgets else [args.budget]

    all_out = {"results": {}}

    for R in budgets:
        out_R = {}

        for snr in snrs:
            res, _ = eval_one(
                baseline_model,
                fis_models,
                channel_type=args.channel,
                snr_db=snr,
                budget=R,
                modes=modes,
                loader=loader,
                device=device,
                seed_base=args.seed,
            )

            out_R[str(snr)] = res

            print(f"\n=== R={R} | SNR={snr} ===")
            for m in modes:
                print(f"{m:16s} PSNR={res[m]['psnr']:.3f} SSIM={res[m]['ssim']:.4f}")

        all_out["results"][str(R)] = out_R

    out_path = os.path.join(args.save_dir, "paper_sims_results.json")
    with open(out_path, "w") as f:
        json.dump(all_out, f, indent=2)

    print("\nSaved:", out_path)


if __name__ == "__main__":
    import random, numpy as np

    seed = 1234
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main()