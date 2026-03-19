import json
import sys
import os

out_path = sys.argv[1]
files = sys.argv[2:]

merged = None

for fpath in files:
    with open(fpath, "r") as f:
        data = json.load(f)

    if merged is None:
        merged = data
    else:
        for R in data["results"]:
            for snr in data["results"][R]:
                for m, v in data["results"][R][snr]["metrics_meanstd"].items():
                    merged["results"][R][snr]["metrics_meanstd"][m] = v

with open(out_path, "w") as f:
    json.dump(merged, f, indent=2)

print("Merged ->", out_path)