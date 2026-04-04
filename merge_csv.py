import pandas as pd
import matplotlib.pyplot as plt
import os

# ====== CONFIG ======
input_csv = "/media/data/students/nguyenquangkhai/metrics_cifar10_noeq_awgn_budget_sweep.csv"
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# ====== LOAD DATA ======
df = pd.read_csv(input_csv)

# ====== SORT DATA ======
df = df.sort_values(by=["budget", "snr_db", "method"])

# ====== EXPORT LATEX TABLE ======
latex_file = os.path.join(output_dir, "table.tex")

with open(latex_file, "w") as f:
    for budget in sorted(df["budget"].unique()):
        sub_df = df[df["budget"] == budget]

        f.write(f"\\begin{{table}}[h]\n\\centering\n")
        f.write(f"\\caption{{Results at budget = {budget}}}\n")
        f.write("\\begin{tabular}{c|ccccc}\n")
        f.write("\\hline\n")
        f.write("SNR & Baseline & Linear & Importance & SNR-only & Full \\\\\n")
        f.write("\\hline\n")

        for snr in sorted(sub_df["snr_db"].unique()):
            row = sub_df[sub_df["snr_db"] == snr]

            def get_val(method):
                val = row[row["method"] == method]["psnr"].values
                return f"{val[0]:.2f}" if len(val) else "-"

            baseline = get_val("baseline")
            linear = get_val("linear")
            importance = get_val("importance_only")
            snr_only = get_val("snr_only")
            full = get_val("full")

            f.write(f"{snr} & {baseline} & {linear} & {importance} & {snr_only} & {full} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n\n")

print(f"LaTeX table saved to {latex_file}")

# ====== PLOT PSNR ======
for budget in sorted(df["budget"].unique()):
    plt.figure()

    sub_df = df[df["budget"] == budget]

    for method in sub_df["method"].unique():
        method_df = sub_df[sub_df["method"] == method]
        plt.plot(method_df["snr_db"], method_df["psnr"], marker='o', label=method)

    plt.xlabel("SNR (dB)")
    plt.ylabel("PSNR")
    plt.title(f"PSNR vs SNR (budget={budget})")
    plt.legend()
    plt.grid()

    save_path = os.path.join(output_dir, f"psnr_budget_{budget}.png")
    plt.savefig(save_path)
    plt.close()

# ====== PLOT SSIM ======
for budget in sorted(df["budget"].unique()):
    plt.figure()

    sub_df = df[df["budget"] == budget]

    for method in sub_df["method"].unique():
        method_df = sub_df[sub_df["method"] == method]
        plt.plot(method_df["snr_db"], method_df["ssim"], marker='o', label=method)

    plt.xlabel("SNR (dB)")
    plt.ylabel("SSIM")
    plt.title(f"SSIM vs SNR (budget={budget})")
    plt.legend()
    plt.grid()

    save_path = os.path.join(output_dir, f"ssim_budget_{budget}.png")
    plt.savefig(save_path)
    plt.close()

print("Plots saved in outputs/")