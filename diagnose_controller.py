import argparse
import json
import os
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import create_dataset
from channel import Channel
from model import DeepJSCC_FIS, power_normalize
from model_baseline import DeepJSCC as DeepJSCC_Baseline, ratio2filtersize


def parse_modes(s: str):
    return [m.strip().lower() for m in s.split(',') if m.strip()]


def tensor_stats(x: torch.Tensor) -> Dict[str, float]:
    x = x.detach().float()
    return {
        'mean': float(x.mean().item()),
        'std': float(x.std(unbiased=False).item()),
        'min': float(x.min().item()),
        'max': float(x.max().item()),
    }


def per_location_energy(z: torch.Tensor) -> torch.Tensor:
    return z.detach().float().pow(2).mean(dim=1)


def flat_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.detach().float().reshape(-1)
    b = b.detach().float().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(eps)
    return float((a @ b / denom).item())


def hist_counts(x: torch.Tensor, bins: int = 12, x_min: float = None, x_max: float = None):
    arr = x.detach().float().cpu().numpy().reshape(-1)
    if x_min is None:
        x_min = float(arr.min())
    if x_max is None:
        x_max = float(arr.max())
    counts, edges = np.histogram(arr, bins=bins, range=(x_min, x_max))
    return {'edges': [float(v) for v in edges.tolist()], 'counts': [int(v) for v in counts.tolist()]}


def load_fis_model(ckpt_path: str, c: int, ratio: float, channel_type: str, rician_k: float, device: torch.device):
    model = DeepJSCC_FIS(c=c, ratio=ratio, channel_type=channel_type, rician_k=rician_k).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def load_baseline_model(ckpt_path: str, c: int, channel_type: str, rician_k: float, device: torch.device):
    model = DeepJSCC_Baseline(c=c, channel_type=channel_type, rician_k=rician_k).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt, strict=False)
    model.eval()
    return model


def _clone_pending_fading(pending):
    if pending is None:
        return None
    kind = pending[0]
    tensors = [t.clone() if torch.is_tensor(t) else t for t in pending[1:]]
    return (kind, *tensors)


@torch.no_grad()
def run_one_mode(model, x, snr_db, budget, mode, channel_ctx):
    z = model.encoder(x)
    A, info = model.controller(
        z,
        snr_db=snr_db,
        budget=budget,
        mode=mode,
        channel_rel=channel_ctx['gamma_eff_norm'],
        return_info=True,
    )
    gain = A.clamp_min(model.eps)
    z_g = z * gain.unsqueeze(1)
    z_tx = power_normalize(z_g, P=model.P, eps=model.eps)
    E_z = per_location_energy(z)
    E_zg = per_location_energy(z_g)
    E_ztx = per_location_energy(z_tx)

    # FIX: gamma_eff_norm is [batch], A is [batch, c, H, W]
    A_mean_per_sample = A.view(A.shape[0], -1).mean(dim=1)

    out = {
        'mode': mode,
        'channel_ctx': {
            'gamma_eff_db_stats': tensor_stats(channel_ctx['gamma_eff_db']),
            'gamma_eff_norm_stats': tensor_stats(channel_ctx['gamma_eff_norm']),
            'h_abs2_stats': tensor_stats(channel_ctx['h_abs2']),
            'posteq_noise_var_stats': tensor_stats(channel_ctx['posteq_noise_var']),
            'eq_quality_stats': tensor_stats(channel_ctx['eq_quality']),
        },
        'I_stats': tensor_stats(info['I']) if 'I' in info else None,
        'channel_rel_stats': tensor_stats(info['channel_rel']) if 'channel_rel' in info else None,
        'A_stats': tensor_stats(A),
        'z_stats': tensor_stats(z),
        'z_g_stats': tensor_stats(z_g),
        'z_tx_stats': tensor_stats(z_tx),
        'E_z_stats': tensor_stats(E_z),
        'Ezg_stats': tensor_stats(E_zg),
        'E_ztx_stats': tensor_stats(E_ztx),
        'A_hist': hist_counts(A, bins=12),
        'E_ztx_hist': hist_counts(E_ztx, bins=12),
        'corr_A_I': flat_corr(A, info['I']) if 'I' in info else None,
        'corr_A_gamma_eff': flat_corr(A_mean_per_sample, channel_ctx['gamma_eff_norm']),
        'mean_power_z': float(z.pow(2).mean().item()),
        'mean_power_z_g': float(z_g.pow(2).mean().item()),
        'mean_power_z_tx': float(z_tx.pow(2).mean().item()),
        'mean_abs_shift_gating': float((z_g - z).abs().mean().item()),
        'mean_abs_shift_tx_vs_pre': float((z_tx - z).abs().mean().item()),
        'tensors': {
            'I': info.get('I', None),
            'channel_rel': info.get('channel_rel', None),
            'A': A,
            'z': z,
            'z_g': z_g,
            'z_tx': z_tx,
        },
    }
    if 'A_raw' in info:
        out['A_raw_stats'] = tensor_stats(info['A_raw'])
    if 'rule1_id' in info:
        ids = info['rule1_id'].detach().cpu().numpy().reshape(-1)
        uniq, cnt = np.unique(ids, return_counts=True)
        out['rule1_usage'] = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
    if 'rule2_id' in info:
        ids = info['rule2_id'].detach().cpu().numpy().reshape(-1)
        uniq, cnt = np.unique(ids, return_counts=True)
        out['rule2_usage'] = {int(k): int(v) for k, v in zip(uniq.tolist(), cnt.tolist())}
    return out


@torch.no_grad()
def run_baseline(model, x):
    z = model.encoder(x)
    z_tx = power_normalize(z, P=1.0, eps=1e-8)
    E_z = per_location_energy(z)
    E_ztx = per_location_energy(z_tx)
    return {
        'mode': 'baseline',
        'z_stats': tensor_stats(z),
        'z_tx_stats': tensor_stats(z_tx),
        'E_z_stats': tensor_stats(E_z),
        'E_ztx_stats': tensor_stats(E_ztx),
        'E_ztx_hist': hist_counts(E_ztx, bins=12),
        'mean_power_z': float(z.pow(2).mean().item()),
        'mean_power_z_tx': float(z_tx.pow(2).mean().item()),
        'tensors': {'z': z, 'z_tx': z_tx},
    }


def save_map_png(x, path, title=''):
    import matplotlib.pyplot as plt
    arr = x.detach().float().cpu().numpy()
    plt.figure(figsize=(4, 4))
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline_ckpt', type=str, default='')
    ap.add_argument('--linear_ckpt', type=str, required=True)
    ap.add_argument('--importance_only_ckpt', type=str, required=True)
    ap.add_argument('--snr_only_ckpt', type=str, required=True)
    ap.add_argument('--full_ckpt', type=str, required=True)
    ap.add_argument('--ratio', type=float, default=1/12)
    ap.add_argument('--channel', type=str, default='AWGN', choices=['AWGN', 'Rayleigh', 'Rician', 'rayleigh_legacy'])
    ap.add_argument('--rician_k', type=float, default=4.0)
    ap.add_argument('--rayleigh_equalize', action='store_true')
    ap.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'celebahq', 'folder'])
    ap.add_argument('--data_root', type=str, default='')
    ap.add_argument('--image_size', type=int, default=32)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--batch_index', type=int, default=0)
    ap.add_argument('--sample_index', type=int, default=0)
    ap.add_argument('--snr_db', type=float, default=1.0)
    ap.add_argument('--budget', type=float, default=1.0)
    ap.add_argument('--save_dir', type=str, default='diag_out')
    ap.add_argument('--modes', type=str, default='linear,importance_only,snr_only,full')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = create_dataset(args.dataset, split='test', data_root=args.data_root, image_size=args.image_size, random_flip=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    x0, _ = ds[0]
    c = ratio2filtersize(x0, args.ratio)

    batch = None
    for i, (x, _) in enumerate(loader):
        if i == args.batch_index:
            batch = x.to(device)
            break
    if batch is None:
        raise ValueError(f'batch_index={args.batch_index} out of range')

    ch = Channel(channel_type=args.channel, snr_db=args.snr_db, rician_k=args.rician_k)
    ch.enable_rayleigh_equalization(args.rayleigh_equalize)
    channel_ctx = ch.sample_context(batch_size=batch.shape[0], device=batch.device, dtype=batch.dtype)

    results = {
        'meta': {
            'channel': args.channel,
            'rician_k': args.rician_k,
            'rayleigh_equalize': bool(args.rayleigh_equalize),
            'dataset': args.dataset,
            'image_size': args.image_size,
            'batch_size': args.batch_size,
            'batch_index': args.batch_index,
            'sample_index': args.sample_index,
            'snr_db': args.snr_db,
            'budget': args.budget,
            'ratio': args.ratio,
            'latent_c': c,
        },
        'shared_channel_ctx': {
            'gamma_eff_db_stats': tensor_stats(channel_ctx['gamma_eff_db']),
            'gamma_eff_norm_stats': tensor_stats(channel_ctx['gamma_eff_norm']),
            'h_abs2_stats': tensor_stats(channel_ctx['h_abs2']),
            'posteq_noise_var_stats': tensor_stats(channel_ctx['posteq_noise_var']),
            'eq_quality_stats': tensor_stats(channel_ctx['eq_quality']),
        },
        'modes': {},
        'comparisons_to_full': {},
    }

    if args.baseline_ckpt:
        baseline = load_baseline_model(args.baseline_ckpt, c, args.channel, args.rician_k, device)
        base_out = run_baseline(baseline, batch)
        results['modes']['baseline'] = {k: v for k, v in base_out.items() if k != 'tensors'}

    mode_to_ckpt = {
        'linear': args.linear_ckpt,
        'importance_only': args.importance_only_ckpt,
        'snr_only': args.snr_only_ckpt,
        'full': args.full_ckpt,
    }

    raw = {}
    for mode in parse_modes(args.modes):
        model = load_fis_model(mode_to_ckpt[mode], c, args.ratio, args.channel, args.rician_k, device)
        out = run_one_mode(model, batch, args.snr_db, args.budget, mode, channel_ctx)
        raw[mode] = out
        results['modes'][mode] = {k: v for k, v in out.items() if k != 'tensors'}

        s = args.sample_index
        tensors = out['tensors']
        if tensors.get('I', None) is not None:
            save_map_png(tensors['I'][s], os.path.join(args.save_dir, f'{mode}_I_sample{s}.png'), f'{mode} I')

        # FIX: skip scalar channel_rel (cannot imshow a scalar)
        if tensors.get('channel_rel', None) is not None:
            _cr = tensors['channel_rel'][s]
            if _cr.dim() >= 2 or (_cr.dim() == 1 and _cr.numel() > 1):
                save_map_png(_cr, os.path.join(args.save_dir, f'{mode}_channelRel_sample{s}.png'), f'{mode} channel_rel')

        save_map_png(tensors['A'][s], os.path.join(args.save_dir, f'{mode}_A_sample{s}.png'), f'{mode} A')
        save_map_png(per_location_energy(tensors['z'])[s], os.path.join(args.save_dir, f'{mode}_Ez_sample{s}.png'), f'{mode} E(z)')
        save_map_png(per_location_energy(tensors['z_g'])[s], os.path.join(args.save_dir, f'{mode}_Ezg_sample{s}.png'), f'{mode} E(z_g)')
        save_map_png(per_location_energy(tensors['z_tx'])[s], os.path.join(args.save_dir, f'{mode}_Eztx_sample{s}.png'), f'{mode} E(z_tx)')

    ref = raw['full']['tensors']
    for mode, out in raw.items():
        t = out['tensors']
        # FIX: gamma_eff shape mismatch
        A_mean = t['A'].view(t['A'].shape[0], -1).mean(dim=1)
        ref_A_mean = ref['A'].view(ref['A'].shape[0], -1).mean(dim=1)
        comp = {
            'A_l1_mean_to_full': float((t['A'] - ref['A']).abs().mean().item()),
            'A_l2_rel_to_full': float((t['A'] - ref['A']).pow(2).mean().sqrt().item() / (ref['A'].pow(2).mean().sqrt().item() + 1e-8)),
            'z_tx_l1_mean_to_full': float((t['z_tx'] - ref['z_tx']).abs().mean().item()),
            'z_tx_l2_rel_to_full': float((t['z_tx'] - ref['z_tx']).pow(2).mean().sqrt().item() / (ref['z_tx'].pow(2).mean().sqrt().item() + 1e-8)),
            'corr_A_with_full_A': flat_corr(t['A'], ref['A']),
            'corr_z_tx_energy_with_full': flat_corr(per_location_energy(t['z_tx']), per_location_energy(ref['z_tx'])),
            'corr_A_with_gamma_eff': flat_corr(A_mean, channel_ctx['gamma_eff_norm']),
        }
        results['comparisons_to_full'][mode] = comp

    out_path = os.path.join(args.save_dir, 'diagnostics.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f'\nSaved diagnostics to: {out_path}')
    print(f'Saved maps to: {args.save_dir}')


if __name__ == '__main__':
    main()