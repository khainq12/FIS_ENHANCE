import json
import os
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIG =================
files = {
    "AWGN": "/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch/paper_sims_awgn/paper_sims_results.json",
    "Rayleigh_EQ": "/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch/paper_sims_eq/paper_sims_results.json",
    "Rayleigh_noEQ": "/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch/paper_sims_noeq/paper_sims_results.json",
}

methods = ["baseline", "linear", "importance_only", "snr_only", "full"]

method_names = {
    "baseline": "Baseline",
    "linear": "Linear",
    "importance_only": "Importance",
    "snr_only": "SNR-only",
    "full": "Full"
}

colors = {
    "baseline": "black",
    "linear": "blue",
    "importance_only": "green",
    "snr_only": "orange",
    "full": "red"
}

linestyles = {
    "baseline": "-",
    "linear": "--",
    "importance_only": "-.",
    "snr_only": ":",
    "full": "-"
}

os.makedirs("plots", exist_ok=True)

# ================= LOAD =================
def load_data(path):
    with open(path, "r") as f:
        return json.load(f)

# ================= EXTRACT RESULTS =================
def extract_results(data):
    results = data["results"]["1.0"]
    snrs = sorted([float(k) for k in results.keys()])

    out = {m: {"psnr": [], "ssim": [], "time": []} for m in methods}

    for snr in snrs:
        entry = results[str(snr)]
        for m in methods:
            out[m]["psnr"].append(entry[m]["psnr"])
            out[m]["ssim"].append(entry[m]["ssim"])
            out[m]["time"].append(entry[m]["time"])

    return snrs, out

# ================= EXTRACT CONTROL =================
def extract_control_stats(data):
    ctrl = data["control_stats"]["1.0"]
    snrs = sorted([float(k) for k in ctrl.keys()])

    metrics = ["A_std", "A_range", "I_A_corr", "channel_rel_mean"]

    out = {m: {metric: [] for metric in metrics} for m in methods}

    for snr in snrs:
        entry = ctrl[str(snr)]
        for m in methods:
            if m in entry:
                for metric in metrics:
                    out[m][metric].append(entry[m][metric])
            else:
                for metric in metrics:
                    out[m][metric].append(0.0)

    return snrs, out

# ================= LATEX =================
def generate_latex(channel_name, snrs, data):
    latex = []

    def build_table(metric, fmt):
        latex.append("\\begin{table}[h]")
        latex.append("\\centering")
        latex.append(f"\\caption{{{channel_name} {metric.upper()}}}")
        latex.append("\\begin{tabular}{c|" + "c"*len(methods) + "}")
        latex.append("\\hline")
        latex.append("SNR & " + " & ".join(method_names[m] for m in methods) + " \\\\ \\hline")

        for i, snr in enumerate(snrs):
            values = [data[m][metric][i] for m in methods]
            best_idx = int(np.argmax(values))

            row = [f"{snr}"]
            for j, m in enumerate(methods):
                val = fmt.format(data[m][metric][i])
                if j == best_idx:
                    val = f"\\textbf{{{val}}}"
                row.append(val)

            latex.append(" & ".join(row) + " \\\\")

        latex.append("\\hline")
        latex.append("\\end{tabular}")
        latex.append("\\end{table}\n")

    build_table("psnr", "{:.2f}")
    build_table("ssim", "{:.3f}")
    build_table("time", "{:.4f}")

    # ===== gain vs baseline =====
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{channel_name} PSNR Gain over Baseline}}")
    latex.append("\\begin{tabular}{c|" + "c"*(len(methods)-1) + "}")
    latex.append("\\hline")
    latex.append("SNR & " + " & ".join(method_names[m] for m in methods if m!="baseline") + " \\\\ \\hline")

    for i, snr in enumerate(snrs):
        base = data["baseline"]["psnr"][i]
        row = [f"{snr}"]

        for m in methods:
            if m == "baseline":
                continue
            gain = data[m]["psnr"][i] - base
            row.append(f"{gain:+.2f}")

        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}\n")

    return "\n".join(latex)

# ================= PLOT =================
def plot_metric(snrs, data, metric, title, save_path):
    plt.figure(figsize=(6,4))

    for m in methods:
        plt.plot(
            snrs,
            data[m][metric],
            marker='o',
            linestyle=linestyles[m],
            color=colors[m],
            label=method_names[m],
            linewidth=2
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ================= CONTROL PLOT =================
def plot_control(snrs, ctrl_data, metric, title, save_path):
    plt.figure(figsize=(6,4))

    for m in methods:
        if m == "baseline":
            continue

        plt.plot(
            snrs,
            ctrl_data[m][metric],
            marker='o',
            linestyle=linestyles[m],
            color=colors[m],
            label=method_names[m],
            linewidth=2
        )

    plt.xlabel("SNR (dB)")
    plt.ylabel(metric)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

# ================= MAIN =================
output_tex = "results_tables.tex"

with open(output_tex, "w") as f_tex:

    for name, path in files.items():
        print(f"Processing {name}...")

        data = load_data(path)

        # ===== results =====
        snrs, extracted = extract_results(data)

        latex_str = generate_latex(name, snrs, extracted)
        f_tex.write(latex_str)

        plot_metric(snrs, extracted, "psnr", f"{name} PSNR", f"plots/{name}_psnr.png")
        plot_metric(snrs, extracted, "ssim", f"{name} SSIM", f"plots/{name}_ssim.png")
        plot_metric(snrs, extracted, "time", f"{name} Time", f"plots/{name}_time.png")

        # ===== control =====
        snrs_ctrl, ctrl = extract_control_stats(data)

        plot_control(snrs_ctrl, ctrl, "A_std", f"{name} A_std", f"plots/{name}_A_std.png")
        plot_control(snrs_ctrl, ctrl, "I_A_corr", f"{name} I-A Corr", f"plots/{name}_I_A_corr.png")
        plot_control(snrs_ctrl, ctrl, "channel_rel_mean", f"{name} Channel Usage", f"plots/{name}_channel.png")

print("Done! All tables + plots generated.")