
import argparse
import json


def build_metric_table(data, budget, methods, snrs, caption_metric, metric_key, fmt, caption_channel):
    res = data["results"][budget]
    lines = []
    lines.append("\\begin{table}[!t]")
    lines.append(f"\\caption{{{caption_metric} on {caption_channel} at $R={budget}$.}}")
    lines.append("\\centering\\footnotesize")
    lines.append("\\begin{tabular}{l" + "c" * len(snrs) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join([f"{s} dB" for s in snrs]) + " \\\\")
    lines.append("\\midrule")

    def get_metric(snr, method, key):
        snr_key = str(float(snr))
        if snr_key not in res:
            return None
        node = res[snr_key]
        if "metrics_meanstd" in node:
            if method not in node["metrics_meanstd"]:
                return None
            return node["metrics_meanstd"][method].get(key, None)
        if "metrics" in node:
            if method not in node["metrics"]:
                return None
            base_key = key.replace("_mean", "")
            return node["metrics"][method].get(base_key, None)
        if method not in node:
            return None
        if key.startswith("psnr"):
            return node[method].get("psnr", None)
        if key.startswith("ssim"):
            return node[method].get("ssim", None)
        if key.startswith("time"):
            return node[method].get("time", None)
        return None

    for method in methods:
        row = []
        for snr in snrs:
            val = get_metric(snr, method, metric_key)
            if val is None:
                row.append("--")
            else:
                row.append(fmt.format(val))
        lines.append(method.replace("_", "\\_") + " & " + " & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=str, required=True)
    ap.add_argument("--budget", type=str, default="1.0")
    ap.add_argument("--channel", type=str, default="")
    ap.add_argument("--out", type=str, default="tables.tex")
    ap.add_argument("--methods", type=str, default="baseline,linear,importance_only,snr_only,full")
    ap.add_argument("--snrs", type=str, default="1,4,7,10,13")
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        data = json.load(f)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    snrs = [s.strip() for s in args.snrs.split(",") if s.strip()]
    channel_name = data.get("channel", "") or args.channel or "the evaluated channel"

    lines = ["% Auto-generated LaTeX tables from paper_sims_results.json"]
    lines += build_metric_table(data, args.budget, methods, snrs, "PSNR (dB)", "psnr_mean", "{:.3f}", channel_name)
    lines += build_metric_table(data, args.budget, methods, snrs, "SSIM", "ssim_mean", "{:.4f}", channel_name)
    lines += build_metric_table(
        data, args.budget, methods, snrs, "Average inference time (s)", "time_mean", "{:.4f}", channel_name
    )

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Wrote", args.out)


if __name__ == "__main__":
    main()
