import json
import pandas as pd
import argparse
import os

# ================== ARGPARSE ==================
parser = argparse.ArgumentParser()
parser.add_argument("--channel", type=str, required=True)
parser.add_argument("--root", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)

# 🔥 NEW: eq / noeq
parser.add_argument("--eq_tag", type=str, default="noeq", choices=["eq", "noeq"])

args = parser.parse_args()

# ================== NORMALIZE CHANNEL ==================
channel_map = {
    "awgn": "AWGN",
    "AWGN": "AWGN",
    "rayleigh": "Rayleigh",
    "Rayleigh": "Rayleigh",
    "rician": "Rician",
    "Rician": "Rician"
}

if args.channel not in channel_map:
    raise ValueError(f"Invalid channel: {args.channel}")

channel = channel_map[args.channel]
root = args.root
output_dir = args.output_dir
eq_tag = args.eq_tag

os.makedirs(output_dir, exist_ok=True)

print(f"Channel: {channel}")
print(f"Setting: {eq_tag}")

# ================== PATH CONFIG ==================
def build_path(method):
    return os.path.join(
        root,
        f"ckpts_{eq_tag}_{method}_{channel}",
        "rule_usage_best.json"
    )

methods = ["linear", "snr_only", "importance_only", "full"]

files = {m: build_path(m) for m in methods}

# ================== LOAD ==================
records = []

for method, path in files.items():
    if not os.path.exists(path):
        print(f"⚠️ Missing file: {path}")
        continue

    print(f"✔ Loading: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    for layer, values in data.items():
        for i, v in enumerate(values):
            records.append({
                "method": method,
                "layer": layer,
                "rule": f"Rule {i}",
                "value": v
            })

# ================== CHECK ==================
df = pd.DataFrame(records)

if df.empty:
    raise RuntimeError("❌ No data loaded. Check checkpoint paths!")

# ================== SORT METHOD ORDER ==================
method_order = ["linear", "importance_only", "snr_only", "full"]
df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)

# ================== PIVOT ==================
pivot = df.pivot_table(
    index="rule",
    columns=["layer", "method"],
    values="value"
)

pivot = pivot.sort_index()

# sort columns nicely
pivot = pivot.sort_index(axis=1, level=[0, 1])

pivot = pivot.round(4)

# ================== EXPORT LATEX ==================
caption = f"Rule usage per layer and method ({channel}, {eq_tag})"
label = f"tab:rule_usage_{channel.lower()}_{eq_tag}"

latex = pivot.to_latex(
    caption=caption,
    label=label,
    float_format="%.4f",
    multicolumn=True,
    multicolumn_format='c',
)

# ================== SAVE ==================
output_path = os.path.join(
    output_dir,
    f"rule_usage_{channel.lower()}_{eq_tag}.tex"
)

with open(output_path, "w") as f:
    f.write(latex)

print(f"\n✅ Saved: {output_path}")