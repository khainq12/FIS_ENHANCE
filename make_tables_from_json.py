import argparse, json, os

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

    res = data["results"][args.budget]

    # ======================
    # SAFE GET (FULL FIX)
    # ======================
    def get_metric(snr, m, key):
        # FIX 1: normalize SNR key (1 -> 1.0)
        snr_key = str(float(snr))

        if snr_key not in res:
            return None

        node = res[snr_key]

        # Case 1: metrics_meanstd
        if "metrics_meanstd" in node:
            if m not in node["metrics_meanstd"]:
                return None
            return node["metrics_meanstd"][m].get(key, None)

        # Case 2: legacy metrics
        if "metrics" in node:
            if m not in node["metrics"]:
                return None
            base_key = key.replace("_mean", "")
            return node["metrics"][m].get(base_key, None)

        # Case 3: DIRECT FORMAT (của bạn)
        if m not in node:
            return None

        if key.startswith("psnr"):
            return node[m].get("psnr", None)

        if key.startswith("ssim"):
            return node[m].get("ssim", None)

        return None

    lines = []
    lines.append("% Auto-generated LaTeX tables from paper_sims_results.json")

    # ======================
    # PSNR TABLE
    # ======================
    lines.append("\\begin{table}[!t]")
    lines.append("\\caption{PSNR (dB) on %s at $R=%s$.}" % (data.get("channel",""), args.budget))
    lines.append("\\centering\\footnotesize")
    lines.append("\\begin{tabular}{l" + "c"*len(snrs) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join([f"{s} dB" for s in snrs]) + " \\\\")
    lines.append("\\midrule")

    for m in methods:
        row = []
        for s in snrs:
            mu = get_metric(s, m, "psnr_mean")
            sd = get_metric(s, m, "psnr_std")

            if mu is None:
                row.append("--")
                continue

            if sd is not None and sd > 0:
                row.append(f"{mu:.3f}$\\pm${sd:.3f}")
            else:
                row.append(f"{mu:.3f}")

        lines.append(m.replace("_","\\_") + " & " + " & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    # ======================
    # SSIM TABLE
    # ======================
    lines.append("\\begin{table}[!t]")
    lines.append("\\caption{SSIM on %s at $R=%s$.}" % (data.get("channel",""), args.budget))
    lines.append("\\centering\\footnotesize")
    lines.append("\\begin{tabular}{l" + "c"*len(snrs) + "}")
    lines.append("\\toprule")
    lines.append("Method & " + " & ".join([f"{s} dB" for s in snrs]) + " \\\\")
    lines.append("\\midrule")

    for m in methods:
        row = []
        for s in snrs:
            mu = get_metric(s, m, "ssim_mean")
            sd = get_metric(s, m, "ssim_std")

            if mu is None:
                row.append("--")
                continue

            if sd is not None and sd > 0:
                row.append(f"{mu:.4f}$\\pm${sd:.4f}")
            else:
                row.append(f"{mu:.4f}")

        lines.append(m.replace("_","\\_") + " & " + " & ".join(row) + " \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("✅ Wrote", args.out)


if __name__ == "__main__":
    main()