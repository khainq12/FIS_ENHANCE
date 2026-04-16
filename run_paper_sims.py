
import argparse
import json
import os
import time
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import create_dataset
from channel import Channel
from model_baseline import DeepJSCC as DeepJSCC_Baseline, ratio2filtersize
from model import DeepJSCC_FIS, power_normalize
from utils import get_psnr, simple_ssim


def _normalize_loaded_state_dict(ckpt):
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):
        raise TypeError("Checkpoint must contain a state_dict-like mapping.")
    out = {}
    for k, v in ckpt.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        out[nk] = v
    return out


def resolve_fis_ckpt_path(mode, channel, fis_ckpt_root, eq_tag="", fis_ckpt_map_json=""):
    if fis_ckpt_map_json:
        with open(fis_ckpt_map_json, "r", encoding="utf-8") as f:
            ckpt_map = json.load(f)
        if mode not in ckpt_map:
            raise KeyError(f"Mode '{mode}' missing from fis_ckpt_map_json.")
        return ckpt_map[mode]
    ckpt_dir = f"ckpts_{eq_tag}_{mode}_{channel}" if eq_tag else f"ckpts_{mode}_{channel}"
    return os.path.join(fis_ckpt_root, ckpt_dir, "fis_power_best.pth")


def _clone_pending_fading(pending):
    if pending is None:
        return None
    kind = pending[0]
    tensors = [t.clone() if torch.is_tensor(t) else deepcopy(t) for t in pending[1:]]
    return (kind, *tensors)


def _ctx_stats(ctx):
    out = {}
    for k in ["gamma_eff_db", "gamma_eff_norm", "h_abs2", "posteq_noise_var", "eq_quality"]:
        if k in ctx:
            x = ctx[k].detach().float()
            out[k] = {
                "mean": float(x.mean().item()),
                "std": float(x.std(unbiased=False).item()),
                "min": float(x.min().item()),
                "max": float(x.max().item()),
            }
    out["channel_type"] = str(ctx.get("channel_type", ""))
    out["fading_equalize"] = bool(float(ctx.get("fading_equalize", torch.tensor(0.0)).item())) if torch.is_tensor(ctx.get("fading_equalize", None)) else bool(ctx.get("fading_equalize", False))
    return out


def _merge_control_stats(acc, info):
    if "A_std" in info:
        acc["A_std"] += float(info["A_std"].mean().item())
    if "A_range" in info:
        acc["A_range"] += float(info["A_range"].mean().item())
    if "I_A_corr" in info:
        acc["I_A_corr"] += float(info["I_A_corr"].mean().item())
    if "channel_rel_mean" in info:
        acc["channel_rel_mean"] += float(info["channel_rel_mean"].mean().item())
    elif "channel_ctx" in info and "channel_rel" in info["channel_ctx"]:
        acc["channel_rel_mean"] += float(info["channel_ctx"]["channel_rel"].float().mean().item())
    if "delta_map" in info:
        acc["delta_mean"] += float(info["delta_map"].mean().item())
    if "score_map" in info:
        acc["score_abs_mean"] += float(info["score_map"].abs().mean().item())


@torch.no_grad()
def eval_one(
    baseline_model,
    fis_models,
    channel_type,
    snr_db,
    budget,
    modes,
    loader,
    device,
    seed_base=1234,
    rayleigh_equalize=False,
    max_batches=None,
    rician_k=4.0,
):
    shared_channel = Channel(channel_type=channel_type, snr_db=snr_db, rician_k=rician_k)
    shared_channel.enable_rayleigh_equalization(rayleigh_equalize)
    shared_channel.change_snr(snr_db)

    results = {m: {"psnr": 0.0, "ssim": 0.0, "n": 0, "time": 0.0} for m in modes}
    control_logs = {
        m: {"A_std": 0.0, "A_range": 0.0, "I_A_corr": 0.0, "channel_rel_mean": 0.0, "delta_mean": 0.0, "score_abs_mean": 0.0, "n": 0}
        for m in modes if m != "baseline"
    }
    ctx_logs = []

    for bidx, (x, _) in enumerate(loader):
        if max_batches is not None and bidx >= max_batches:
            break
        x = x.to(device)
        B = x.shape[0]
        torch.manual_seed(seed_base + bidx)
        np.random.seed(seed_base + bidx)

        # Sample one shared channel context / fading realization for this batch.
        ctx = shared_channel.sample_context(batch_size=B, device=x.device, dtype=x.dtype)
        pending = _clone_pending_fading(shared_channel._pending_fading)
        ctx_logs.append(_ctx_stats(ctx))

        # ----- baseline -----
        t0 = time.time()
        z = baseline_model.encoder(x)
        z = power_normalize(z, P=1.0, eps=1e-8)
        shared_channel._pending_fading = _clone_pending_fading(pending)
        z_noisy = shared_channel(z)
        x_hat = baseline_model.decoder(z_noisy)
        elapsed = time.time() - t0

        results["baseline"]["time"] += elapsed
        results["baseline"]["psnr"] += get_psnr(x_hat * 255.0, x * 255.0, max_val=255.0).mean().item()
        results["baseline"]["ssim"] += simple_ssim(x_hat, x).mean().item()
        results["baseline"]["n"] += 1

        # ----- FIS modes -----
        for mode in modes:
            if mode == "baseline":
                continue
            fis_model = fis_models[mode]

            t0 = time.time()
            z = fis_model.encoder(x)
            A, info = fis_model.controller(
                z,
                snr_db=snr_db,
                budget=budget,
                mode=mode,
                channel_rel=ctx.get("channel_rel", ctx["gamma_eff_norm"]),
                return_info=True,
            )
            _merge_control_stats(control_logs[mode], info)
            control_logs[mode]["n"] += 1
            z_g = z * A.unsqueeze(1)
            z_tx = power_normalize(z_g, P=fis_model.P, eps=fis_model.eps)
            shared_channel._pending_fading = _clone_pending_fading(pending)
            z_noisy = shared_channel(z_tx)
            x_hat = fis_model.decoder(z_noisy)
            elapsed = time.time() - t0

            results[mode]["time"] += elapsed
            results[mode]["psnr"] += get_psnr(x_hat * 255.0, x * 255.0, max_val=255.0).mean().item()
            results[mode]["ssim"] += simple_ssim(x_hat, x).mean().item()
            results[mode]["n"] += 1

    for mode in modes:
        n = max(results[mode]["n"], 1)
        results[mode]["psnr"] /= n
        results[mode]["ssim"] /= n
        results[mode]["time"] /= n

    # Aggregate channel-context logs across batches.
    agg = {}
    if ctx_logs:
        keys = [k for k in ctx_logs[0].keys() if isinstance(ctx_logs[0][k], dict)]
        for key in keys:
            means = [d[key]["mean"] for d in ctx_logs]
            stds = [d[key]["std"] for d in ctx_logs]
            mins = [d[key]["min"] for d in ctx_logs]
            maxs = [d[key]["max"] for d in ctx_logs]
            agg[key] = {
                "mean_of_means": float(np.mean(means)),
                "mean_of_stds": float(np.mean(stds)),
                "global_min": float(np.min(mins)),
                "global_max": float(np.max(maxs)),
            }
        agg["channel_type"] = ctx_logs[0].get("channel_type", channel_type)
        agg["fading_equalize"] = bool(ctx_logs[0].get("fading_equalize", False))

    control_agg = {}
    for mode, stats in control_logs.items():
        n = max(int(stats.pop("n", 0)), 1)
        control_agg[mode] = {k: float(v / n) for k, v in stats.items()}

    return results, agg, control_agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_ckpt", type=str, required=True)
    ap.add_argument("--fis_ckpt_root", type=str, default="")
    ap.add_argument("--fis_ckpt_map_json", type=str, default="")
    ap.add_argument("--eq_tag", type=str, default="")
    ap.add_argument("--rayleigh_equalize", action="store_true")
    ap.add_argument("--ratio", type=float, default=1/6)
    ap.add_argument("--channel", type=str, default="AWGN",
                    choices=["AWGN", "Rayleigh", "Rician", "rayleigh_legacy"])
    ap.add_argument("--rician_k", type=float, default=4.0)
    ap.add_argument("--snrs", type=float, nargs='+', default=[1, 4, 7, 10, 13])
    ap.add_argument("--budget", type=float, default=1.0)
    ap.add_argument("--budgets", type=float, nargs='+', default=None)
    ap.add_argument("--modes", type=str, default="baseline,linear,importance_only,snr_only,full")
    ap.add_argument("--dataset", type=str, default="cifar10")
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--image_size", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_dir", type=str, default="paper_sims_out")
    ap.add_argument("--snr_only_from_baseline", action="store_true",
                    help="Treat snr_only as a no-op diagnostic using the baseline backbone instead of a separately trained ckpt.")
    args = ap.parse_args()
    if "snr_only" in [m.strip().lower() for m in args.modes.split(",") if m.strip()] and args.baseline_ckpt and not args.snr_only_from_baseline:
        args.snr_only_from_baseline = True
        print("[INFO] snr_only detected -> auto-enabling baseline-backed no-op diagnostic in evaluation.")

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    testset = create_dataset(
        args.dataset, split="test", data_root=args.data_root,
        image_size=args.image_size, random_flip=False
    )
    loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    x0, _ = testset[0]
    c = ratio2filtersize(x0, args.ratio)

    baseline_model = DeepJSCC_Baseline(
        c=c, channel_type=args.channel, rician_k=args.rician_k
    ).to(device)
    baseline_model.load_state_dict(torch.load(args.baseline_ckpt, map_location=device), strict=True)
    baseline_model.eval()

    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    if "snr_only" in modes:
        print("[WARN] snr_only is an identity spatial controller in the current implementation.")
        print("[WARN] Treat it as a no-spatial-control diagnostic, not as true SNR-aware allocation.")
    fis_models = {}
    loaded_fis_paths = {}

    for mode in modes:
        if mode == "baseline":
            continue
        model = DeepJSCC_FIS(
            c=c, ratio=args.ratio, channel_type=args.channel, rician_k=args.rician_k
        ).to(device)
        if mode == "snr_only" and args.snr_only_from_baseline:
            print("Loading snr_only from baseline backbone (no-op diagnostic):", args.baseline_ckpt)
            baseline_sd = _normalize_loaded_state_dict(torch.load(args.baseline_ckpt, map_location=device))
            missing, unexpected = model.load_state_dict(baseline_sd, strict=False)
            print("  strict=False load for snr_only baseline init | missing:", len(missing), "unexpected:", len(unexpected))
            loaded_fis_paths[mode] = args.baseline_ckpt
        else:
            ckpt_path = resolve_fis_ckpt_path(
                mode=mode,
                channel=args.channel,
                fis_ckpt_root=args.fis_ckpt_root,
                eq_tag=args.eq_tag,
                fis_ckpt_map_json=args.fis_ckpt_map_json,
            )
            print("Loading:", ckpt_path)
            model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
            loaded_fis_paths[mode] = ckpt_path
        model.eval()
        fis_models[mode] = model

    snrs = args.snrs
    budgets = args.budgets if args.budgets is not None else [args.budget]

    all_out = {
        "channel": args.channel,
        "dataset": args.dataset,
        "image_size": args.image_size,
        "rayleigh_equalize": bool(args.rayleigh_equalize),
        "baseline_ckpt": args.baseline_ckpt,
        "fis_ckpts": loaded_fis_paths,
        "results": {},
        "channel_context": {},
        "control_stats": {},
    }

    for R in budgets:
        out_R = {}
        ctx_R = {}
        for snr in snrs:
            res, ctx_stats, ctrl_stats = eval_one(
                baseline_model,
                fis_models,
                channel_type=args.channel,
                snr_db=snr,
                budget=R,
                modes=modes,
                loader=loader,
                device=device,
                rayleigh_equalize=args.rayleigh_equalize,
                rician_k=args.rician_k,
            )
            out_R[str(snr)] = res
            ctx_R[str(snr)] = ctx_stats
            out_R.setdefault("_control", {})[str(snr)] = ctrl_stats
            print(f"\n=== R={R} | SNR={snr} ===")
            for m in modes:
                print(
                    f"{m:16s} PSNR={res[m]['psnr']:.3f} "
                    f"SSIM={res[m]['ssim']:.4f} time={res[m]['time']:.4f}s"
                )
            if ctx_stats:
                gedb = ctx_stats.get("gamma_eff_db", {})
                print(f"channel_ctx gamma_eff_db mean={gedb.get('mean_of_means', float('nan')):.3f} std={gedb.get('mean_of_stds', float('nan')):.3f}")
        control_R = out_R.pop("_control", {})
        all_out["results"][str(R)] = out_R
        all_out["channel_context"][str(R)] = ctx_R
        all_out["control_stats"][str(R)] = control_R

    out_path = os.path.join(args.save_dir, "paper_sims_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2)
    print("\nSaved:", out_path)


if __name__ == "__main__":
    import random

    seed = 1234
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    main()
