# -*- coding: utf-8 -*-
"""
Train DeepJSCC + FIS power controller.

Fixes vs repo train_fis_power.py:
---------------------------------
1) Import from fixed modules (channel_fixed, model_fixed, fis_modules_fixed)
2) ★ NEW: Rule balance loss added to training objective
   - lambda_rule_balance controls weight (default 0.01)
   - Encourages all fuzzy rules to fire equally
   - Fixes "rule 4 dominates 40%" problem
3) Per-subcarrier fading is automatic (set in model_fixed.py via c // 2)
4) save_dir default changed to avoid overwriting repo experiments
"""
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


# ═══════════════════════════════════════════════════════════════
# Helpers (same as repo)
# ═══════════════════════════════════════════════════════════════

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


def compute_control_stats(model, loader, device, snr, budget, mode, max_batches=8):
    model.eval()
    agg = {
        "A_std": 0.0, "A_range": 0.0, "I_A_corr": 0.0,
        "channel_rel_mean": 0.0, "delta_mean": 0.0, "score_abs_mean": 0.0,
    }
    count = 0
    with torch.no_grad():
        for bidx, (x, _) in enumerate(loader):
            if max_batches is not None and bidx >= max_batches:
                break
            x = x.to(device)
            _, _, info = model(x, snr=snr, budget=budget, mode=mode, return_info=True)
            if "A_std" in info:
                agg["A_std"] += float(info["A_std"].mean().item())
            if "A_range" in info:
                agg["A_range"] += float(info["A_range"].mean().item())
            if "I_A_corr" in info:
                agg["I_A_corr"] += float(info["I_A_corr"].mean().item())
            if "channel_rel_mean" in info:
                agg["channel_rel_mean"] += float(info["channel_rel_mean"].mean().item())
            elif "channel_ctx" in info and "channel_rel" in info["channel_ctx"]:
                agg["channel_rel_mean"] += float(
                    info["channel_ctx"]["channel_rel"].float().mean().item()
                )
            if "delta_map" in info:
                agg["delta_mean"] += float(info["delta_map"].mean().item())
            if "score_map" in info:
                agg["score_abs_mean"] += float(info["score_map"].abs().mean().item())
            count += 1
    if count == 0:
        return {k: 0.0 for k in agg}
    return {k: v / count for k, v in agg.items()}


def set_requires_grad(module: nn.Module, flag: bool) -> None:
    for p in module.parameters():
        p.requires_grad = flag


def _normalize_loaded_state_dict(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must contain a state_dict-like mapping.")
    out = {}
    for k, v in ckpt.items():
        nk = k[7:] if k.startswith("module.") else k
        out[nk] = v
    return out


def load_backbone_from_baseline(fis_model: nn.Module, ckpt_path: str,
                                device: torch.device) -> nn.Module:
    """Load baseline weights into FIS model (encoder + decoder only).

    ★ FIX: When loading from a block-fading baseline into a per-subcarrier
    model, num_fading_taps changes but encoder/decoder weights are identical.
    This is safe because the encoder/decoder don't depend on fading taps.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    src_sd = _normalize_loaded_state_dict(ckpt)
    dst_sd = fis_model.state_dict()

    copied = []
    for k, v in src_sd.items():
        if k in dst_sd and dst_sd[k].shape == v.shape:
            dst_sd[k] = v
            copied.append(k)

    fis_model.load_state_dict(dst_sd, strict=False)
    print(f"[Warm-start] Copied {len(copied)} tensors from baseline: {ckpt_path}")
    if len(copied) < len(src_sd):
        skipped = [k for k in src_sd if k not in copied and not k.startswith("module.")]
        print(f"[Warm-start] Skipped {len(skipped)} mismatched keys: {skipped[:5]}...")
    return fis_model


@torch.no_grad()
def evaluate_multi_snr(model, loader, device, snr_list, budget, mode,
                       channel, rician_k):
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


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

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
    ap.add_argument("--save_dir", type=str, default="ckpts_fis_fixed")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--random_flip", action="store_true")
    ap.add_argument("--rayleigh_equalize", action="store_true")
    ap.add_argument("--train_snr_list", type=float, nargs="+", default=None)
    ap.add_argument("--eval_snr_list", type=float, nargs="+", default=None)
    ap.add_argument("--baseline_ckpt", type=str, default="")
    ap.add_argument("--warmstart_controller_only_epochs", type=int, default=10)
    ap.add_argument("--finetune_lr", type=float, default=1e-5)
    ap.add_argument("--snr_only_use_baseline_backbone", action="store_true")

    # ★ FIX ISSUE 7: NEW — rule balance loss weight
    ap.add_argument("--lambda_rule_balance", type=float, default=0.01,
                    help="Weight for KL rule-balance loss (0 = disabled).")

    args = ap.parse_args()

    if args.mode == "snr_only" and args.baseline_ckpt and not args.snr_only_use_baseline_backbone:
        args.snr_only_use_baseline_backbone = True
        print("[INFO] mode=snr_only with baseline_ckpt -> auto-enabling no-op export.")
    if args.mode == "snr_only" and not args.snr_only_use_baseline_backbone:
        print("[WARN] snr_only without --snr_only_use_baseline_backbone will confound ablation.")

    os.makedirs(args.save_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainset = create_dataset(
        args.dataset, split="train", data_root=args.data_root,
        image_size=args.image_size, random_flip=args.random_flip,
    )
    testset = create_dataset(
        args.dataset, split="test", data_root=args.data_root,
        image_size=args.image_size, random_flip=False,
    )
    train_loader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    image_first = trainset[0][0]
    c = ratio2filtersize(image_first, args.ratio)

    train_snr_list = (
        parse_snr_list(args.train_snr_list) if args.train_snr_list is not None
        else [args.snr_min, 0.5 * (args.snr_min + args.snr_max), args.snr_max]
    )
    eval_snr_list = (
        parse_snr_list(args.eval_snr_list) if args.eval_snr_list is not None
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
        "baseline_ckpt": args.baseline_ckpt,
        "warmstart_controller_only_epochs": args.warmstart_controller_only_epochs,
        "finetune_lr": args.finetune_lr,
        "lambda_rule_balance": args.lambda_rule_balance,
        "fix_applied": [
            "per_subcarrier_fading",
            "strengthened_option_c",
            "spatial_noise_in_channel_conditioning",
            "rule_balance_loss",
            "bug2_syntax_fix",
            "bug4_alpha_linear_learnable",
            "bug_legacy_yQ_fix",
            "bypass_threshold_adjusted",
        ],
    }
    with open(os.path.join(args.save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ── Build model (uses per-subcarrier fading automatically) ──
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

    num_taps = model.channel.num_fading_taps
    print(f"[INFO] Encoder channels c={c}, num_fading_taps={num_taps}")
    print(f"[INFO] channel_rel shape will be (B, {num_taps}) for fading channels")

    if args.baseline_ckpt:
        model = load_backbone_from_baseline(model, args.baseline_ckpt, device)

    # ── snr_only shortcut ──
    if args.mode == "snr_only" and args.baseline_ckpt and args.snr_only_use_baseline_backbone:
        print("[snr_only] No-op controller on baseline backbone. No training.")
        psnr_avg = evaluate_multi_snr(
            model, test_loader, device, snr_list=eval_snr_list,
            budget=args.budget, mode=args.mode,
            channel=args.channel, rician_k=args.rician_k,
        )
        torch.save(model.state_dict(), os.path.join(args.save_dir, "fis_power_best.pth"))
        with open(os.path.join(args.save_dir, "rule_usage_best.json"), "w") as f:
            json.dump({}, f, indent=2)
        snr_diag = float(sum(eval_snr_list) / len(eval_snr_list))
        ctrl_stats = compute_control_stats(
            model, test_loader, device, snr=snr_diag,
            budget=args.budget, mode=args.mode,
        )
        with open(os.path.join(args.save_dir, "control_stats_best.json"), "w") as f:
            json.dump(ctrl_stats, f, indent=2)
        print(f"[snr_only] mean PSNR over {eval_snr_list} dB = {psnr_avg:.3f}")
        return

    # ── Optimizer setup ──
    controller_has_params = sum(1 for _ in model.controller.parameters()) > 0

    if args.baseline_ckpt and controller_has_params:
        set_requires_grad(model.encoder, False)
        set_requires_grad(model.decoder, False)
        set_requires_grad(model.controller, True)
        opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
        )
        n_ctrl = sum(1 for p in model.controller.parameters())
        print(f"[Warm-start] Phase 1: controller-only ({n_ctrl} params)")
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    loss_fn = nn.MSELoss()
    best_psnr = -1.0
    phase2_switched = False
    use_two_phase = args.baseline_ckpt and controller_has_params

    lambda_rule = args.lambda_rule_balance  # ★ FIX ISSUE 7

    print(f"Training SNR list: {train_snr_list}")
    print(f"Eval SNR list: {eval_snr_list}")
    print(f"Rule balance lambda: {lambda_rule}")

    # ── Training loop ──
    for ep in range(1, args.epochs + 1):
        # Phase switch
        if (
            use_two_phase and not phase2_switched
            and ep == args.warmstart_controller_only_epochs + 1
        ):
            set_requires_grad(model.encoder, True)
            set_requires_grad(model.decoder, True)
            set_requires_grad(model.controller, True)
            opt = torch.optim.Adam(model.parameters(), lr=args.finetune_lr)
            phase2_switched = True
            print("[Phase switch] Unfroze encoder+decoder, lr -> finetune_lr")

        model.train()
        t0 = time.time()
        running_mse = 0.0
        running_rule = 0.0
        count = 0

        for x, _ in train_loader:
            x = x.to(device)
            snr = random.choice(train_snr_list)
            model.set_channel(channel_type=args.channel, snr=snr, rician_k=args.rician_k)

            # ★ FIX ISSUE 7: Always use return_info to get rule balance loss
            _, x_hat, info = model(
                x, snr=snr, budget=args.budget, mode=args.mode, return_info=True
            )

            mse_loss = loss_fn(x_hat, x)

            # ★ FIX ISSUE 7: Add rule balance loss
            rule_bal = torch.tensor(0.0, device=device)
            if lambda_rule > 0:
                if "rule1_balance_loss" in info:
                    rule_bal = rule_bal + info["rule1_balance_loss"]
                if "rule2_balance_loss" in info:
                    rule_bal = rule_bal + info["rule2_balance_loss"]

            loss = mse_loss + lambda_rule * rule_bal

            opt.zero_grad()
            loss.backward()
            opt.step()

            running_mse += mse_loss.item()
            running_rule += rule_bal.item()
            count += 1

        avg_mse = running_mse / max(count, 1)
        avg_rule = running_rule / max(count, 1)

        # ── Evaluate ──
        psnr_avg = evaluate_multi_snr(
            model, test_loader, device, snr_list=eval_snr_list,
            budget=args.budget, mode=args.mode,
            channel=args.channel, rician_k=args.rician_k,
        )

        new_best = False
        if psnr_avg > best_psnr:
            best_psnr = psnr_avg
            new_best = True
            torch.save(model.state_dict(), os.path.join(args.save_dir, "fis_power_best.pth"))
            print(f">>> New BEST (mean PSNR={psnr_avg:.3f}) -> saving rule usage...")
            snr_diag = float(sum(eval_snr_list) / len(eval_snr_list))
            rule_usage = compute_rule_usage(
                model, test_loader, device, snr=snr_diag,
                budget=args.budget, mode=args.mode,
            )
            with open(os.path.join(args.save_dir, "rule_usage_best.json"), "w") as f:
                json.dump(rule_usage, f, indent=2)
            ctrl_stats = compute_control_stats(
                model, test_loader, device, snr=snr_diag,
                budget=args.budget, mode=args.mode,
            )
            with open(os.path.join(args.save_dir, "control_stats_best.json"), "w") as f:
                json.dump(ctrl_stats, f, indent=2)
            print(f"    control_stats @ {snr_diag:.2f} dB: {ctrl_stats}")

        torch.save(model.state_dict(), os.path.join(args.save_dir, f"fis_power_ep{ep:03d}.pth"))
        print(
            f"Epoch {ep:03d} | MSE={avg_mse:.6f} | rule_bal={avg_rule:.4f} | "
            f"mean_PSNR={psnr_avg:.3f} dB | time={time.time() - t0:.1f}s"
            f"{' >>> BEST' if new_best else ''}"
        )

    print(f"Best mean PSNR: {best_psnr:.3f}")


if __name__ == "__main__":
    main()
