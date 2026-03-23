import json
import glob
import csv
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--pattern", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

pattern = os.path.join(args.input_dir, args.pattern)
files = glob.glob(pattern)

if not files:
    raise ValueError(f"No files found: {pattern}")

rows = []

for fpath in files:
    print("Reading:", fpath)

    # lấy budget từ folder name
    folder = os.path.basename(os.path.dirname(fpath))
    budget = float(folder.split("B")[-1])

    with open(fpath, "r") as f:
        data = json.load(f)

    results = data["results"]

    for budget_key in results:         # "0.5"
        for snr_key in results[budget_key]:   # "1.0", "4.0", ...

            methods = results[budget_key][snr_key]

            for method, val in methods.items():

                psnr = val["psnr"]
                ssim = val["ssim"]

                rows.append({
                    "snr_db": float(snr_key),
                    "budget": float(budget_key),
                    "method": method,
                    "psnr": psnr,
                    "ssim": ssim
                })

# sort cho đẹp
rows = sorted(rows, key=lambda x: (x["budget"], x["snr_db"], x["method"]))

# ghi CSV
with open(args.output, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["snr_db", "budget", "method", "psnr", "ssim"]
    )
    writer.writeheader()
    writer.writerows(rows)

print("========================================")
print("Saved CSV ->", args.output)
print(f"Total rows: {len(rows)}")
print("========================================")