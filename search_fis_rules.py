import argparse
import itertools
import json
import os
import statistics
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset import Vanilla
from model import DeepJSCC_FIS, power_normalize
from model_baseline import DeepJSCC as DeepJSCC_Baseline, ratio2filtersize
from channel import Channel
from utils import get_psnr


def parse_list(s, cast=float):
    return [cast(x.strip()) for x in s.split(',') if x.strip()]


@torch.no_grad()
def eval_config(baseline_model, fis_model, loader, device, channel_type, snrs, budget, max_batches=None, rayleigh_equalize=False):
    modes = ["baseline", "snr_only", "importance_only", "full"]
    out = {m: [] for m in modes}

    for snr_db in snrs:
        channel = Channel(channel_type=channel_type, snr_db=snr_db)
        if hasattr(channel, 'enable_rayleigh_equalization'):
            channel.enable_rayleigh_equalization(rayleigh_equalize)
        if hasattr(channel, 'change_snr'):
            channel.change_snr(snr_db)

        psnr_acc = {m: 0.0 for m in modes}
        n = 0
        for bidx, (x, _) in enumerate(loader):
            if max_batches is not None and bidx >= max_batches:
                break
            x = x.to(device)

            z = baseline_model.encoder(x)
            z = power_normalize(z, P=1.0, eps=fis_model.eps)
            x_hat = baseline_model.decoder(channel(z))
            psnr_acc['baseline'] += get_psnr(x_hat * 255.0, x * 255.0, max_val=255.0)

            zf = fis_model.encoder(x)
            for mode in ["snr_only", "importance_only", "full"]:
                A = fis_model.controller(zf, snr_db=snr_db, budget=budget, mode=mode, return_info=False)
                gain = torch.sqrt(A.clamp_min(fis_model.eps))
                z_tx = power_normalize(zf * gain.unsqueeze(1), P=fis_model.P, eps=fis_model.eps)
                x_hat = fis_model.decoder(channel(z_tx))
                psnr_acc[mode] += get_psnr(x_hat * 255.0, x * 255.0, max_val=255.0)
            n += 1

        for m in modes:
            out[m].append(psnr_acc[m] / max(n, 1))
    return out


def score_result(res, objective='full_vs_all'):
    full = res['full']
    baseline = res['baseline']
    snr_only = res['snr_only']
    imp_only = res['importance_only']
    deltas_base = [f - b for f, b in zip(full, baseline)]
    deltas_snr = [f - s for f, s in zip(full, snr_only)]
    deltas_imp = [f - i for f, i in zip(full, imp_only)]

    if objective == 'full_vs_baseline':
        return statistics.mean(deltas_base)
    if objective == 'full_vs_snr':
        return statistics.mean(deltas_snr)

    # Strongly prefer full >= baseline everywhere, then improve against snr_only/importance_only.
    penalty = 0.0
    for d in deltas_base:
        if d < 0:
            penalty += 5.0 * abs(d)
    for d in deltas_snr:
        if d < -0.02:
            penalty += 2.0 * abs(d + 0.02)
    for d in deltas_imp:
        if d < -0.01:
            penalty += 1.5 * abs(d + 0.01)

    return (
        1.0 * statistics.mean(deltas_base)
        + 0.8 * statistics.mean(deltas_snr)
        + 0.4 * statistics.mean(deltas_imp)
        - penalty
    )


def apply_config(fis_model, cfg):
    fis_model.controller.pow.c = torch.tensor(cfg['pow_c'], dtype=torch.float32)
    fis_model.controller.pow.w0 = float(cfg['w0'])
    fis_model.controller.smooth_kernel = int(cfg['smooth_kernel'])
    fis_model.controller.alpha_linear = float(cfg['alpha_linear'])
    return fis_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline_ckpt', type=str, default='')
    ap.add_argument('--fis_ckpt', type=str, required=True)
    ap.add_argument('--ratio', type=float, default=1/12)
    ap.add_argument('--channel', type=str, default='AWGN', choices=['AWGN', 'Rayleigh', 'rayleigh_legacy'])
    ap.add_argument('--snrs', type=str, default='1,4,7,10,13')
    ap.add_argument('--budget', type=float, default=1.0)
    ap.add_argument('--batch_size', type=int, default=128)
    ap.add_argument('--num_workers', type=int, default=2)
    ap.add_argument('--max_batches', type=int, default=20)
    ap.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'folder'])
    ap.add_argument('--data_root', type=str, default='')
    ap.add_argument('--resize', type=int, default=0)
    ap.add_argument('--save_dir', type=str, default='fis_rule_search_out')
    ap.add_argument('--objective', type=str, default='full_vs_all', choices=['full_vs_all', 'full_vs_baseline', 'full_vs_snr'])
    ap.add_argument('--rayleigh_equalize', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.resize and args.resize > 0:
        tfm = transforms.Compose([transforms.Resize(args.resize), transforms.CenterCrop(args.resize), transforms.ToTensor()])
    else:
        tfm = transforms.ToTensor()

    if args.dataset == 'cifar10':
        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tfm)
    else:
        if not args.data_root:
            raise ValueError('--data_root is required when --dataset folder')
        testset = Vanilla(root=args.data_root, transform=tfm)
    loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    x0, _ = testset[0]
    if x0.dim() == 3:
        x0 = x0.unsqueeze(0)
    c = ratio2filtersize(x0, args.ratio)

    baseline_model = DeepJSCC_Baseline(c=c, channel_type=args.channel).to(device)
    fis_model = DeepJSCC_FIS(ratio=args.ratio, channel_type=args.channel).to(device)

    if args.baseline_ckpt:
        baseline_model.load_state_dict(torch.load(args.baseline_ckpt, map_location=device), strict=False)
    fis_model.load_state_dict(torch.load(args.fis_ckpt, map_location=device), strict=False)
    baseline_model.eval()
    fis_model.eval()

    snrs = parse_list(args.snrs, float)

    # 48 configs: 4 rule packs x 3 w0 x 2 smooth x 2 alpha.
    pow_c_grid = [
        [1.45, 1.30, 1.10, 1.05, 1.00, 0.85],
        [1.36, 1.24, 1.08, 1.04, 1.00, 0.90],
        [1.30, 1.20, 1.08, 1.04, 1.00, 0.92],
        [1.24, 1.16, 1.06, 1.03, 1.00, 0.95],
    ]
    w0_grid = [0.03, 0.05, 0.08]
    smooth_grid = [1, 3]
    alpha_grid = [0.6, 0.8]

    best = None
    trials = []
    idx = 0
    for pow_c, w0, smooth_kernel, alpha_linear in itertools.product(pow_c_grid, w0_grid, smooth_grid, alpha_grid):
        idx += 1
        cfg = {
            'pow_c': pow_c,
            'w0': w0,
            'smooth_kernel': smooth_kernel,
            'alpha_linear': alpha_linear,
        }
        apply_config(fis_model, cfg)
        res = eval_config(
            baseline_model=baseline_model,
            fis_model=fis_model,
            loader=loader,
            device=device,
            channel_type=args.channel,
            snrs=snrs,
            budget=args.budget,
            max_batches=args.max_batches,
            rayleigh_equalize=args.rayleigh_equalize,
        )
        score = score_result(res, objective=args.objective)
        row = {'trial': idx, 'config': deepcopy(cfg), 'score': score, 'result': res}
        trials.append(row)

        mean_full = statistics.mean(res['full'])
        mean_base = statistics.mean(res['baseline'])
        mean_snr = statistics.mean(res['snr_only'])
        print(f"[{idx:02d}/48] score={score:+.4f} | full={mean_full:.3f} | base={mean_base:.3f} | snr_only={mean_snr:.3f} | cfg={cfg}")

        if best is None or score > best['score']:
            best = row
            with open(os.path.join(args.save_dir, 'best_fis_config.json'), 'w', encoding='utf-8') as f:
                json.dump(best, f, indent=2)

    trials_sorted = sorted(trials, key=lambda x: x['score'], reverse=True)
    with open(os.path.join(args.save_dir, 'all_trials.json'), 'w', encoding='utf-8') as f:
        json.dump(trials_sorted, f, indent=2)

    print('\nBest config:')
    print(json.dumps(best, indent=2))
    print(f"\nSaved to {args.save_dir}")


if __name__ == '__main__':
    main()
