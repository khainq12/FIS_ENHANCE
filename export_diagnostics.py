import json
import os
import matplotlib.pyplot as plt

# ====== INPUT ======
files = {
    "AWGN": "/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch/diag_awgn_snr1/diagnostics.json",
    "Rayleigh_EQ": "/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch/diag_eq_snr1/diagnostics.json",
    "Rayleigh_noEQ": "/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch/diag_noeq_snr1/diagnostics.json",
}

modes = ["baseline", "linear", "importance_only", "snr_only", "full"]

os.makedirs("diag_plots", exist_ok=True)

# ====== LOAD ======
def load(path):
    with open(path, "r") as f:
        return json.load(f)

# ====== PLOT HIST ======
def plot_hist(edges, counts, title, save_path):
    centers = [(edges[i] + edges[i+1]) / 2 for i in range(len(edges)-1)]

    plt.figure()
    plt.bar(centers, counts, width=(edges[1] - edges[0]))
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.grid()

    plt.savefig(save_path)
    plt.close()

# ====== BAR COMPARISON ======
def plot_bar(values_dict, title, save_path):
    names = list(values_dict.keys())
    values = list(values_dict.values())

    plt.figure()
    plt.bar(names, values)
    plt.title(title)
    plt.xticks(rotation=30)
    plt.grid()

    plt.savefig(save_path)
    plt.close()

# ====== LATEX ======
def generate_latex(channel, data):
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Diagnostics ({channel}, SNR=1dB)}}")
    lines.append("\\begin{tabular}{c|c|c|c}")
    lines.append("\\hline")
    lines.append("Mode & mean A & corr(A,I) & mean power z \\\\ \\hline")

    for m in modes:
        mode = data["modes"][m]

        A_mean = mode.get("A_stats", {}).get("mean", 1.0)
        corr = mode.get("corr_A_I", 0.0)
        power = mode.get("mean_power_z", 0.0)

        lines.append(f"{m} & {A_mean:.3f} & {corr:.3f} & {power:.1f} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}\n")

    return "\n".join(lines)

# ====== MAIN ======
with open("diagnostics_tables.tex", "w") as f_tex:

    for name, path in files.items():
        data = load(path)

        # ===== HISTOGRAM =====
        for m in modes:
            mode = data["modes"][m]

            # A histogram
            if "A_hist" in mode:
                plot_hist(
                    mode["A_hist"]["edges"],
                    mode["A_hist"]["counts"],
                    f"{name} - {m} A distribution",
                    f"diag_plots/{name}_{m}_A.png"
                )

            # Energy histogram
            if "E_ztx_hist" in mode:
                plot_hist(
                    mode["E_ztx_hist"]["edges"],
                    mode["E_ztx_hist"]["counts"],
                    f"{name} - {m} Energy distribution",
                    f"diag_plots/{name}_{m}_E.png"
                )

        # ===== BAR CHART =====
        corr_dict = {}
        power_dict = {}

        for m in modes:
            mode = data["modes"][m]
            corr_dict[m] = mode.get("corr_A_I", 0.0)
            power_dict[m] = mode.get("mean_power_z", 0.0)

        plot_bar(corr_dict, f"{name} corr(A,I)", f"diag_plots/{name}_corr.png")
        plot_bar(power_dict, f"{name} mean power z", f"diag_plots/{name}_power.png")

        # ===== LATEX =====
        latex = generate_latex(name, data)
        f_tex.write(latex)

print("Done diagnostics!")