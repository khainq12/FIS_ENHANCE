#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diag_budget.py — Budget diagnostic & mode explanation tool.

Two missions:
  1. BUDGET SENSITIVITY: For each budget in [0.0, 0.25, 0.5, 0.75, 1.0],
     log A stats, score_raw vs score_centered, E_top-20%, entropy.
     If A_std ~ 0 across all budgets → budget is dead.
  2. MODE COMPARISON: For each mode (full, snr_only, importance_only, linear),
     log rule usage distribution, rule entropy, A-I correlation,
     and pairwise A correlation between modes.
     If corr(A_full, A_snr_only) > 0.95 → Full ≈ SNR-only.

Outputs:
  - {save_dir}/budget_diag.json       — full numeric diagnostics per budget
  - {save_dir}/mode_comparison.json   — rule usage + entropy per mode
  - {save_dir}/heatmaps/              — A, I, score_raw heatmaps per budget/mode
  - {save_dir}/summary.txt            — human-readable summary

Usage:
  python diag_budget.py \
    --fis_ckpt ckpts_fis_power/fis_power_best.pth \
    --channel AWGN \
    --save_dir diag_out_awgn
"""

import argparse
import json
import math
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import create_dataset
from model import DeepJSCC_FIS, power_normalize
from model_baseline import ratio2filtersize
from fis_modules import (
    FIS_Importance,
    FIS_PowerAllocation,
    _minmax_norm,
    _mf_low,
    _mf_med,
    _mf_high,
    _fuzzy_and,
    _fuzzy_or,
    _mean_normalize,
)


# ============================================================
# Helper: tensor statistics
# ============================================================

def tensor_stats(t: torch.Tensor) -> Dict[str, float]:
    t = t.detach().float()
    return {
        "mean": float(t.mean().item()),
        "std":  float(t.std(unbiased=False).item()),
        "min":  float(t.min().item()),
        "max":  float(t.max().item()),
        "q10":  float(torch.quantile(t.reshape(-1), 0.10).item()),
        "q25":  float(torch.quantile(t.reshape(-1), 0.25).item()),
        "q50":  float(torch.quantile(t.reshape(-1), 0.50).item()),
        "q75":  float(torch.quantile(t.reshape(-1), 0.75).item()),
        "q90":  float(torch.quantile(t.reshape(-1), 0.90).item()),
    }


def histogram_entropy(t: torch.Tensor, bins: int = 50) -> float:
    """Shannon entropy (bits) of a tensor's value distribution."""
    arr = t.detach().float().cpu().numpy().ravel()
    hist, _ = np.histogram(arr, bins=bins, density=True)
    hist = hist[hist > 1e-12]
    return float(-np.sum(hist * np.log2(hist)))


def flat_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float((a @ b / denom).item())


def rule_entropy(rule_strength: torch.Tensor, eps: float = 1e-8) -> float:
    """Entropy (bits) of rule activation distribution."""
    p = rule_strength.detach().float().mean(dim=(0, 2, 3))
    p = p / (p.sum().clamp_min(eps))
    return float(-(p * torch.log2(p.clamp_min(eps))).sum().item())


# ============================================================
# Core: replicate Layer-2 forward to capture score_raw
# ============================================================

@torch.no_grad()
def compute_layer2_score_raw(
    pow_layer: FIS_PowerAllocation,
    I: torch.Tensor,
    snr_db: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate FIS_PowerAllocation.forward() to get:
      - score_raw  : before centering
      - score_post : after centering + score_scale
      - rules      : normalized rule strength [B, 6, H, W]
    """
    s = pow_layer._snr_unit(snr_db, device=I.device, dtype=I.dtype)
    S = s.view(1, 1, 1).expand_as(I)

    iL, iM, iH = _mf_low(I), _mf_med(I), _mf_high(I)
    sL, sM, sH = _mf_low(S), _mf_med(S), _mf_high(S)

    r0 = _fuzzy_and(iH, sL)
    r1 = _fuzzy_and(iH, sM)
    r2 = _fuzzy_and(iH, sH)
    r3 = _fuzzy_and(iM, sL)
    r4 = _fuzzy_and(iM, _fuzzy_or(sL, sM))
    r5 = _fuzzy_and(iL, _fuzzy_or(sL, sM))

    rules_raw = torch.stack([r0, r1, r2, r3, r4, r5], dim=1)
    rules = pow_layer._normalize_rules(rules_raw)

    c = pow_layer.c.to(I.device, I.dtype).view(1, -1, 1, 1)
    score_raw = (rules * c).sum(dim=1)

    score_post = score_raw - score_raw.mean(dim=(1, 2), keepdim=True)
    score_post = pow_layer.score_scale * score_post

    return score_raw, score_post, rules


@torch.no_grad()
def compute_layer1_features(
    imp_layer: FIS_Importance,
    z: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate FIS_Importance.forward() to get:
      - I      : importance map [B, H, W]
      - rules  : rule strength [B, 7, H, W]
      - features: (m, v, e) normalized
    """
    m = z.abs().mean(dim=1)
    v = z.var(dim=1, unbiased=False)
    e = imp_layer._edge_from_m(m)

    m_n = _minmax_norm(m, imp_layer.eps)
    v_n = _minmax_norm(v, imp_layer.eps)
    e_n = _minmax_norm(e, imp_layer.eps)

    mL, mM, mH = _mf_low(m_n), _mf_med(m_n), _mf_high(m_n)
    vL, vM, vH = _mf_low(v_n), _mf_med(v_n), _mf_high(v_n)
    eL, eM, eH = _mf_low(e_n), _mf_med(e_n), _mf_high(e_n)

    r0 = _fuzzy_and(mH, vH)
    r1 = _fuzzy_and(mH, eH)
    r2 = _fuzzy_and(mM, _fuzzy_or(vH, eH))
    r3 = _fuzzy_and(mM, vM, eM)
    r4 = _fuzzy_and(mL, vL, eL)
    r5 = _fuzzy_and(mL, _fuzzy_or(vH, eH))
    r6 = _fuzzy_and(mH, vL, eL)

    rules_raw = torch.stack([r0, r1, r2, r3, r4, r5, r6], dim=1)
    rules = imp_layer._normalize_rules(rules_raw)

    c = imp_layer.c.to(z.device, z.dtype).view(1, -1, 1, 1)
    I = (rules * c).sum(dim=1).clamp(0.0, 1.0)

    return I, rules, (m_n, v_n, e_n)


# ============================================================
# Mission 1: Budget sweep
# ============================================================

def budget_sweep(
    model: DeepJSCC_FIS,
    x: torch.Tensor,
    budgets: List[float],
    snr_db: float,
    mode: str,
    sample_idx: int,
    heatmap_dir: str,
) -> Dict[str, dict]:
    """Run controller across budgets, return per-budget diagnostics."""
    z = model.encoder(x)
    pow_layer = model.controller.pow
    imp_layer = model.controller.imp

    # Get I once (same for all budgets in "full" mode)
    if mode == "snr_only":
        I = torch.full(
            (z.shape[0], z.shape[2], z.shape[3]), 0.5,
            device=z.device, dtype=z.dtype,
        )
    else:
        I, rule1, _ = compute_layer1_features(imp_layer, z)

    # Compute score_raw/score_post once (they don't depend on budget)
    score_raw, score_post, rule2 = compute_layer2_score_raw(pow_layer, I, snr_db)

    # Compute A_fis once (before budget interpolation)
    s = pow_layer._snr_unit(snr_db, device=I.device, dtype=I.dtype)
    delta = pow_layer._delta_from_snr(float(s))
    A_fis = torch.exp(delta * torch.tanh(score_post))
    A_fis = A_fis.clamp(min=pow_layer.a_min, max=pow_layer.a_high)

    results = {}

    for R in budgets:
        # Budget interpolation
        A = float(R) * A_fis + (1.0 - float(R))
        A = _mean_normalize(A, eps=model.eps)

        # Apply gain
        z_g = z * A.unsqueeze(1)
        z_tx = power_normalize(z_g, P=model.P, eps=model.eps)

        # Energy map
        E = z_tx.detach().float().pow(2).mean(dim=1)  # [B, H, W]
        E_flat = E.reshape(-1)

        # Top-20% energy ratio
        top20_thresh = torch.quantile(E_flat, 0.80)
        mask_top20 = E > top20_thresh
        E_top20_ratio = float(
            (E[mask_top20].sum() / (E.sum().clamp_min(model.eps))).item()
        )

        # Dynamic range of A: max/min ratio
        A_range = float(A.max().item() / (A.min().item() + 1e-8))

        # Effective redistribution: std of A after budget interp
        # Compare with A_fis std
        A_fis_std = float(A_fis.std(unbiased=False).item())
        A_std = float(A.std(unbiased=False).item())
        budget_efficiency = A_std / (A_fis_std + 1e-8)

        entry = {
            "budget": R,
            "A_stats": tensor_stats(A),
            "A_fis_stats": tensor_stats(A_fis),
            "A_entropy": histogram_entropy(A),
            "A_fis_entropy": histogram_entropy(A_fis),
            "A_range_max_div_min": A_range,
            "A_std_ratio": budget_efficiency,  # A_std / A_fis_std
            "E_ztx_stats": tensor_stats(E),
            "E_top20_ratio": E_top20_ratio,
            "corr_A_I": flat_corr(A, I) if mode != "snr_only" else None,
            "score_raw_stats": tensor_stats(score_raw),
            "score_post_stats": tensor_stats(score_post),
            "score_centering_loss": float(
                (score_raw.std(unbiased=False) - score_post.std(unbiased=False)).item()
            ),
        }

        # Rule entropy
        if mode != "snr_only" and rule1 is not None:
            entry["rule1_entropy"] = rule_entropy(rule1)
            p1 = rule1.mean(dim=(0, 2, 3))
            p1 = p1 / p1.sum()
            entry["rule1_distribution"] = [round(float(v), 6) for v in p1.tolist()]
        if rule2 is not None:
            entry["rule2_entropy"] = rule_entropy(rule2)
            p2 = rule2.mean(dim=(0, 2, 3))
            p2 = p2 / p2.sum()
            entry["rule2_distribution"] = [round(float(v), 6) for v in p2.tolist()]

        # Quantile spread of A: (q90 - q10) / mean
        a = A.reshape(-1)
        entry["A_spread_ratio"] = float(
            (torch.quantile(a, 0.90) - torch.quantile(a, 0.10)).item()
            / (a.mean().item() + 1e-8)
        )

        results[str(R)] = entry

        # Save heatmaps
        save_heatmaps(
            A=A, I=I, score_raw=score_raw, score_post=score_post,
            E=E, budget=R, mode=mode, snr_db=snr_db,
            sample_idx=sample_idx, save_dir=heatmap_dir,
        )

    return results


# ============================================================
# Mission 2: Mode comparison (at fixed budget=1.0)
# ============================================================

def mode_comparison(
    model: DeepJSCC_FIS,
    x: torch.Tensor,
    modes: List[str],
    snr_db: float,
    budget: float,
    sample_idx: int,
    heatmap_dir: str,
) -> Dict[str, dict]:
    """Compare modes: rule usage, entropy, A-I correlation, A-A correlation."""
    z = model.encoder(x)
    pow_layer = model.controller.pow
    imp_layer = model.controller.imp

    all_A = {}
    all_info = {}

    for mode in modes:
        if mode == "snr_only":
            I = torch.full(
                (z.shape[0], z.shape[2], z.shape[3]), 0.5,
                device=z.device, dtype=z.dtype,
            )
            rule1 = None
        else:
            I, rule1, _ = compute_layer1_features(imp_layer, z)

        snr_use = 10.0 if mode == "importance_only" else float(snr_db)

        if mode == "linear":
            I_c = I - I.mean(dim=(1, 2), keepdim=True)
            s = pow_layer._snr_unit(snr_use, device=I.device, dtype=I.dtype)
            amp = pow_layer._delta_from_snr(float(s))
            score = model.controller.alpha_linear * I_c
            A_fis = torch.exp(amp * torch.tanh(score))
            A_fis = A_fis.clamp(min=model.controller.a_min, max=model.controller.a_high)
            A = float(budget) * A_fis + (1.0 - float(budget))
            A = _mean_normalize(A, eps=model.eps)
            rule2 = None
            score_raw = score
            score_post = score
        else:
            score_raw, score_post, rule2 = compute_layer2_score_raw(pow_layer, I, snr_use)
            s = pow_layer._snr_unit(snr_use, device=I.device, dtype=I.dtype)
            delta = pow_layer._delta_from_snr(float(s))
            A_fis = torch.exp(delta * torch.tanh(score_post))
            A_fis = A_fis.clamp(min=pow_layer.a_min, max=pow_layer.a_high)
            A = float(budget) * A_fis + (1.0 - float(budget))
            A = _mean_normalize(A, eps=model.eps)

        all_A[mode] = A
        all_info[mode] = {
            "A_stats": tensor_stats(A),
            "A_entropy": histogram_entropy(A),
            "corr_A_I": flat_corr(A, I) if mode != "snr_only" else None,
            "score_raw_stats": tensor_stats(score_raw),
            "score_post_stats": tensor_stats(score_post),
        }

        if rule1 is not None:
            all_info[mode]["rule1_entropy"] = rule_entropy(rule1)
            p1 = rule1.mean(dim=(0, 2, 3))
            p1 = p1 / p1.sum()
            all_info[mode]["rule1_distribution"] = [round(float(v), 6) for v in p1.tolist()]
        if rule2 is not None:
            all_info[mode]["rule2_entropy"] = rule_entropy(rule2)
            p2 = rule2.mean(dim=(0, 2, 3))
            p2 = p2 / p2.sum()
            all_info[mode]["rule2_distribution"] = [round(float(v), 6) for v in p2.tolist()]

        # Energy
        z_g = z * A.unsqueeze(1)
        z_tx = power_normalize(z_g, P=model.P, eps=model.eps)
        E = z_tx.detach().float().pow(2).mean(dim=1)
        E_flat = E.reshape(-1)
        top20_thresh = torch.quantile(E_flat, 0.80)
        mask_top20 = E > top20_thresh
        all_info[mode]["E_top20_ratio"] = float(
            (E[mask_top20].sum() / (E.sum().clamp_min(model.eps))).item()
        )

        # PSNR
        from utils import get_psnr
        model.set_channel(snr=snr_db)
        y = model.channel(z_tx)
        x_hat = model.decoder(y)
        psnr = get_psnr(x_hat * 255.0, x * 255.0, max_val=255.0).mean().item()
        all_info[mode]["psnr"] = round(float(psnr), 4)

        # Save heatmaps
        save_heatmaps(
            A=A, I=I, score_raw=score_raw, score_post=score_post,
            E=E, budget=budget, mode=mode, snr_db=snr_db,
            sample_idx=sample_idx, save_dir=heatmap_dir,
        )

    # Pairwise A correlation between modes
    pairwise = {}
    mode_list = list(all_A.keys())
    for i in range(len(mode_list)):
        for j in range(i + 1, len(mode_list)):
            m1, m2 = mode_list[i], mode_list[j]
            corr = flat_corr(all_A[m1], all_A[m2])
            l2_rel = float(
                (all_A[m1] - all_A[m2]).pow(2).mean().sqrt().item()
                / (all_A[m1].pow(2).mean().sqrt().item() + 1e-8)
            )
            key = f"{m1}_vs_{m2}"
            pairwise[key] = {
                "corr_A": round(corr, 6),
                "l2_rel_A": round(l2_rel, 6),
            }

    all_info["__pairwise_corr__"] = pairwise
    return all_info


# ============================================================
# Heatmap saver
# ============================================================

def save_heatmaps(
    A, I, score_raw, score_post, E,
    budget, mode, snr_db, sample_idx, save_dir,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(save_dir, exist_ok=True)
    s = sample_idx

    # Single row: I | score_raw | score_post | A | E
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    # I
    if I is not None:
        arr = I[s].detach().cpu().numpy()
        im0 = axes[0].imshow(arr, cmap="hot", vmin=0, vmax=1)
        axes[0].set_title(f"I  [{arr.min():.3f}, {arr.max():.3f}]")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
    else:
        axes[0].set_title("I (N/A)")

    # score_raw
    arr = score_raw[s].detach().cpu().numpy()
    im1 = axes[1].imshow(arr, cmap="RdBu_r")
    axes[1].set_title(f"score_raw\n[{arr.min():.4f}, {arr.max():.4f}]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # score_post
    arr = score_post[s].detach().cpu().numpy()
    im2 = axes[2].imshow(arr, cmap="RdBu_r")
    axes[2].set_title(f"score_post\n[{arr.min():.4f}, {arr.max():.4f}]")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # A
    arr = A[s].detach().cpu().numpy()
    im3 = axes[3].imshow(arr, cmap="viridis")
    axes[3].set_title(f"A (budget={budget:.2f})\n[{arr.min():.4f}, {arr.max():.4f}]")
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    # E
    arr = E[s].detach().cpu().numpy()
    im4 = axes[4].imshow(arr, cmap="inferno")
    axes[4].set_title(f"E(z_tx)\n[{arr.min():.4f}, {arr.max():.4f}]")
    plt.colorbar(im4, ax=axes[4], fraction=0.046)

    fig.suptitle(f"mode={mode} | SNR={snr_db} dB | budget={budget:.2f} | sample={s}", fontsize=13)
    plt.tight_layout()
    fname = f"{mode}_snr{snr_db}_budget{budget:.2f}_sample{s}.png"
    plt.savefig(os.path.join(save_dir, fname), dpi=160, bbox_inches="tight")
    plt.close()


# ============================================================
# Summary printer
# ============================================================

def print_budget_summary(budget_results: dict, snr_db: float):
    """Print human-readable budget sweep summary."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"BUDGET DIAGNOSTIC SUMMARY  |  SNR = {snr_db} dB")
    lines.append("=" * 80)
    lines.append("")

    header = (f"{'Budget':>8s} | {'A_std':>10s} | {'A_fis_std':>10s} | "
              f"{'A_range':>8s} | {'A_spread':>8s} | {'A_ent':>6s} | "
              f"{'E_top20':>8s} | {'corr(A,I)':>9s} | "
              f"{'score_raw_std':>13s} | {'score_post_std':>14s} | {'center_loss':>12s}")
    lines.append(header)
    lines.append("-" * len(header))

    for b_str, d in budget_results.items():
        A_st = d["A_stats"]
        Af_st = d.get("A_fis_stats", d["A_stats"])
        sr_st = d.get("score_raw_stats", {})
        sp_st = d.get("score_post_stats", {})
        cl = d.get("score_centering_loss", 0.0)
        corr_ai = d.get("corr_A_I", None)
        corr_ai_s = f"{corr_ai:9.4f}" if corr_ai is not None else "     N/A"

        line = (f"{b_str:>8s} | {A_st['std']:10.6f} | {Af_st['std']:10.6f} | "
                f"{d['A_range_max_div_min']:8.4f} | {d['A_spread_ratio']:8.4f} | "
                f"{d['A_entropy']:6.3f} | {d['E_top20_ratio']:8.4f} | "
                f"{corr_ai_s} | "
                f"{sr_st.get('std', 0.0):13.6f} | {sp_st.get('std', 0.0):14.6f} | "
                f"{cl:12.6f}")
        lines.append(line)

    # Verdict
    lines.append("")
    lines.append("-" * len(header))
    stds = [d["A_stats"]["std"] for d in budget_results.values()]
    max_std_diff = max(stds) - min(stds)
    lines.append(f"  A_std range across budgets: [{min(stds):.6f}, {max(stds):.6f}]  "
                 f"(max_delta = {max_std_diff:.6f})")

    if max_std_diff < 0.005:
        lines.append("  >> VERDICT: Budget is DEAD — A_std does not change with budget.")
    elif max_std_diff < 0.02:
        lines.append("  >> VERDICT: Budget is WEAK — A_std barely changes with budget.")
    else:
        lines.append("  >> VERDICT: Budget is ACTIVE — A_std changes with budget.")

    # Check centering loss
    for b_str, d in budget_results.items():
        cl = d.get("score_centering_loss", 0.0)
        sr_std = d.get("score_raw_stats", {}).get("std", 0.0)
        sp_std = d.get("score_post_stats", {}).get("std", 0.0)
        if sr_std > 0.01 and cl / sr_std > 0.5:
            lines.append(f"  >> WARNING at budget={b_str}: centering removes "
                         f"{cl/sr_std*100:.1f}% of score variance "
                         f"(raw_std={sr_std:.6f} -> post_std={sp_std:.6f})")
            break

    lines.append("")
    return "\n".join(lines)


def print_mode_summary(mode_results: dict, snr_db: float):
    """Print human-readable mode comparison summary."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"MODE COMPARISON SUMMARY  |  SNR = {snr_db} dB  |  budget = 1.0")
    lines.append("=" * 80)
    lines.append("")

    modes = [k for k in mode_results if not k.startswith("__")]

    header = (f"{'Mode':>18s} | {'PSNR':>7s} | {'A_std':>10s} | {'A_ent':>6s} | "
              f"{'E_top20':>8s} | {'corr(A,I)':>9s} | "
              f"{'L1_ent':>6s} | {'L2_ent':>6s} | {'score_raw_std':>13s}")
    lines.append(header)
    lines.append("-" * len(header))

    for m in modes:
        d = mode_results[m]
        sr_st = d.get("score_raw_stats", {})
        corr_ai = d.get("corr_A_I", None)
        corr_ai_s = f"{corr_ai:9.4f}" if corr_ai is not None else "     N/A"
        l1_ent = f"{d.get('rule1_entropy', 0.0):6.3f}" if "rule1_entropy" in d else "    N/A"
        l2_ent = f"{d.get('rule2_entropy', 0.0):6.3f}" if "rule2_entropy" in d else "    N/A"

        line = (f"{m:>18s} | {d.get('psnr', 0.0):7.3f} | "
                f"{d['A_stats']['std']:10.6f} | {d['A_entropy']:6.3f} | "
                f"{d.get('E_top20_ratio', 0.0):8.4f} | {corr_ai_s} | "
                f"{l1_ent} | {l2_ent} | "
                f"{sr_st.get('std', 0.0):13.6f}")
        lines.append(line)

    # Pairwise correlation
    pairwise = mode_results.get("__pairwise_corr__", {})
    if pairwise:
        lines.append("")
        lines.append("Pairwise A correlation between modes:")
        lines.append("-" * 50)
        for key, val in sorted(pairwise.items()):
            flag = ""
            if "full_vs_snr" in key and val["corr_A"] > 0.95:
                flag = "  << NEAR-IDENTICAL!"
            elif "full_vs_importance" in key and val["corr_A"] > 0.95:
                flag = "  << NEAR-IDENTICAL!"
            lines.append(f"  {key:>30s}: corr={val['corr_A']:.4f}  l2_rel={val['l2_rel_A']:.4f}{flag}")

    lines.append("")

    # Rule distribution detail
    for m in modes:
        d = mode_results[m]
        if "rule1_distribution" in d:
            dist = d["rule1_distribution"]
            lines.append(f"  {m} Layer-1 rule dist: [{', '.join(f'{v:.4f}' for v in dist)}]")
        if "rule2_distribution" in d:
            dist = d["rule2_distribution"]
            lines.append(f"  {m} Layer-2 rule dist: [{', '.join(f'{v:.4f}' for v in dist)}]")

    lines.append("")
    return "\n".join(lines)


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Budget sensitivity diagnostic & mode explanation tool"
    )
    ap.add_argument("--fis_ckpt", type=str, required=True,
                    help="Path to FIS model checkpoint (fis_power_best.pth)")
    ap.add_argument("--ratio", type=float, default=1/6)
    ap.add_argument("--channel", type=str, default="AWGN",
                    choices=["AWGN", "Rayleigh", "Rician", "rayleigh_legacy"])
    ap.add_argument("--rician_k", type=float, default=4.0)
    ap.add_argument("--dataset", type=str, default="cifar10",
                    choices=["cifar10", "celebahq", "folder"])
    ap.add_argument("--data_root", type=str, default="")
    ap.add_argument("--image_size", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--sample_index", type=int, default=0,
                    help="Which sample in batch to use for heatmaps")
    ap.add_argument("--snr_list", type=float, nargs="+", default=[1.0, 7.0, 13.0],
                    help="SNRs to evaluate")
    ap.add_argument("--budget_list", type=float, nargs="+",
                    default=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0],
                    help="Budgets to sweep")
    ap.add_argument("--modes", type=str, default="full,snr_only,importance_only,linear",
                    help="Modes for comparison (comma-separated)")
    ap.add_argument("--rayleigh_equalize", action="store_true")
    ap.add_argument("--save_dir", type=str, default="diag_budget_out")
    ap.add_argument("--no_heatmaps", action="store_true",
                    help="Skip heatmap generation (faster)")
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    heatmap_base = os.path.join(args.save_dir, "heatmaps")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load dataset & model ----
    ds = create_dataset(
        args.dataset, split="test", data_root=args.data_root,
        image_size=args.image_size, random_flip=False,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    x_batch, _ = next(iter(loader))
    x_batch = x_batch.to(device)

    x0, _ = ds[0]
    c = ratio2filtersize(x0, args.ratio)

    model = DeepJSCC_FIS(
        c=c, ratio=args.ratio, channel_type=args.channel, rician_k=args.rician_k,
    ).to(device)
    model.load_state_dict(torch.load(args.fis_ckpt, map_location=device), strict=False)
    model.eval()
    if hasattr(model, "channel") and hasattr(model.channel, "enable_rayleigh_equalization"):
        model.channel.enable_rayleigh_equalization(args.rayleigh_equalize)

    modes = [m.strip().lower() for m in args.modes.split(",") if m.strip()]
    all_text = []

    # ---- Print model config ----
    ctrl = model.controller
    pow_l = ctrl.pow
    imp_l = ctrl.imp
    all_text.append(f"Model loaded from: {args.fis_ckpt}")
    all_text.append(f"Channel: {args.channel}  |  ratio: {args.ratio}  |  c: {c}")
    all_text.append(f"Layer-2 consequents: {pow_l.c.tolist()}")
    all_text.append(f"  a_min={pow_l.a_min}  a_med={pow_l.a_med}  a_high={pow_l.a_high}")
    all_text.append(f"  score_scale={pow_l.score_scale}  "
                    f"delta_min={pow_l.delta_min}  delta_max={pow_l.delta_max}")
    all_text.append(f"  snr_range=[{pow_l.snr_min_db}, {pow_l.snr_max_db}]")
    all_text.append(f"  rule_temp={pow_l.rule_temp}  rule_floor={pow_l.rule_floor}")
    all_text.append(f"Layer-1 consequents: {imp_l.c.tolist()}")
    all_text.append(f"  rule_temp={imp_l.rule_temp}  rule_floor={imp_l.rule_floor}")
    all_text.append("")

    # ---- Run for each SNR ----
    for snr_db in args.snr_list:
        all_text.append(f"\n{'#' * 80}")
        all_text.append(f"# SNR = {snr_db} dB")
        all_text.append(f"{'#' * 80}\n")

        # --- Mission 1: Budget sweep (mode=full) ---
        budget_results = budget_sweep(
            model=model, x=x_batch,
            budgets=args.budget_list,
            snr_db=snr_db, mode="full",
            sample_idx=args.sample_index,
            heatmap_dir=heatmap_base if not args.no_heatmaps else "/dev/null",
        )
        budget_path = os.path.join(args.save_dir, f"budget_diag_snr{snr_db}.json")
        with open(budget_path, "w", encoding="utf-8") as f:
            json.dump(budget_results, f, indent=2)
        all_text.append(f"Saved: {budget_path}")
        all_text.append("")

        summary = print_budget_summary(budget_results, snr_db)
        all_text.append(summary)
        print(summary)

        # --- Mission 2: Mode comparison (budget=1.0) ---
        mode_results = mode_comparison(
            model=model, x=x_batch,
            modes=modes, snr_db=snr_db, budget=1.0,
            sample_idx=args.sample_index,
            heatmap_dir=heatmap_base if not args.no_heatmaps else "/dev/null",
        )
        mode_path = os.path.join(args.save_dir, f"mode_comparison_snr{snr_db}.json")
        with open(mode_path, "w", encoding="utf-8") as f:
            json.dump(mode_results, f, indent=2)
        all_text.append(f"Saved: {mode_path}")
        all_text.append("")

        summary = print_mode_summary(mode_results, snr_db)
        all_text.append(summary)
        print(summary)

    # ---- Save combined summary ----
    summary_path = os.path.join(args.save_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_text))
    all_text.append(f"\nAll outputs saved to: {args.save_dir}")
    all_text.append(f"Summary text saved to: {summary_path}")

    final = "\n".join(all_text)
    print(final)

    # ---- Also write a compact JSON with verdicts ----
    verdicts = {"channel": args.channel, "snrs": {}}
    for snr_db in args.snr_list:
        bp = os.path.join(args.save_dir, f"budget_diag_snr{snr_db}.json")
        mp = os.path.join(args.save_dir, f"mode_comparison_snr{snr_db}.json")
        if not os.path.exists(bp) or not os.path.exists(mp):
            continue
        with open(bp) as f:
            bdata = json.load(f)
        with open(mp) as f:
            mdata = json.load(f)

        stds = [d["A_stats"]["std"] for d in bdata.values()]
        max_std_diff = max(stds) - min(stds)

        pw = mdata.get("__pairwise_corr__", {})
        corr_full_snr = pw.get("full_vs_snr_only", {}).get("corr_A", 0.0)

        # Get PSNR for each mode
        mode_psnr = {}
        for m in ["full", "snr_only", "importance_only", "linear"]:
            if m in mdata:
                mode_psnr[m] = mdata[m].get("psnr", 0.0)

        verdicts["snrs"][str(snr_db)] = {
            "budget_verdict": "DEAD" if max_std_diff < 0.005 else (
                "WEAK" if max_std_diff < 0.02 else "ACTIVE"
            ),
            "budget_A_std_range": [min(stds), max(stds)],
            "budget_A_std_delta": max_std_diff,
            "corr_full_vs_snr_only": corr_full_snr,
            "full_approx_snr_only": corr_full_snr > 0.95,
            "mode_psnr": mode_psnr,
            "full_minus_snr_only": round(
                mode_psnr.get("full", 0.0) - mode_psnr.get("snr_only", 0.0), 4
            ),
        }

    verdict_path = os.path.join(args.save_dir, "verdicts.json")
    with open(verdict_path, "w", encoding="utf-8") as f:
        json.dump(verdicts, f, indent=2)
    print(f"\nVerdicts saved to: {verdict_path}")


if __name__ == "__main__":
    main()
