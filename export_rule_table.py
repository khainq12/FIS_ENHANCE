
import argparse
import json
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", type=str, required=True)
    parser.add_argument("--root", type=str, default="")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--eq_tag", type=str, default="noeq", choices=["eq", "noeq", ""])
    parser.add_argument("--rule_usage_map_json", type=str, default="",
                        help="Optional JSON mapping method -> explicit rule_usage_best.json path.")
    args = parser.parse_args()

    channel_map = {
        "awgn": "AWGN",
        "AWGN": "AWGN",
        "rayleigh": "Rayleigh",
        "Rayleigh": "Rayleigh",
        "rician": "Rician",
        "Rician": "Rician",
    }
    if args.channel not in channel_map:
        raise ValueError(f"Invalid channel: {args.channel}")

    channel = channel_map[args.channel]
    os.makedirs(args.output_dir, exist_ok=True)

    methods = ["linear", "snr_only", "importance_only", "full"]

    if args.rule_usage_map_json:
        with open(args.rule_usage_map_json, "r", encoding="utf-8") as f:
            files = json.load(f)
    else:
        def build_path(method):
            if args.eq_tag:
                return os.path.join(args.root, f"ckpts_{args.eq_tag}_{method}_{channel}", "rule_usage_best.json")
            return os.path.join(args.root, f"ckpts_{method}_{channel}", "rule_usage_best.json")
        files = {m: build_path(m) for m in methods}

    records = []
    for method in methods:
        path = files.get(method, "")
        if not path or not os.path.exists(path):
            print(f"Missing file for {method}: {path}")
            continue
        print(f"Loading: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for layer, values in data.items():
            for i, v in enumerate(values):
                records.append({
                    "method": method,
                    "layer": layer,
                    "rule": f"Rule {i}",
                    "value": v,
                })

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("No data loaded. Check explicit paths or checkpoint naming.")

    method_order = ["linear", "importance_only", "snr_only", "full"]
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)

    pivot = df.pivot_table(index="rule", columns=["layer", "method"], values="value")
    pivot = pivot.sort_index().sort_index(axis=1, level=[0, 1]).round(4)

    caption_eq = args.eq_tag if args.eq_tag else "default"
    caption = f"Rule usage per layer and method ({channel}, {caption_eq})"
    label = f"tab:rule_usage_{channel.lower()}_{caption_eq}"
    latex = pivot.to_latex(
        caption=caption,
        label=label,
        float_format="%.4f",
        multicolumn=True,
        multicolumn_format="c",
    )

    suffix = caption_eq
    output_path = os.path.join(args.output_dir, f"rule_usage_{channel.lower()}_{suffix}.tex")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(latex)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
