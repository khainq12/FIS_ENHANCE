import json
import os
import matplotlib.pyplot as plt

# ====== INPUT FILES ======
files = {
    "AWGN": "/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch/paper_sims_awgn/paper_sims_results.json",
    "Rayleigh_EQ": "/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch/paper_sims_eq/paper_sims_results.json",
    "Rayleigh_noEQ": "/media/data/students/nguyenquangkhai/project1/JSCC_FIS/Deep-JSCC-PyTorch/paper_sims_noeq/paper_sims_results.json",
}

methods = ["baseline", "linear", "importance_only", "snr_only", "full"]

# ====== LOAD DATA ======
def load_data(path):
    with open(path, "r") as f:
        return json.load(f)

# ====== EXTRACT ======
def extract_results(data):
    results = data["results"]["1.0"]  # fixed outer key
    snrs = sorted([float(k) for k in results.keys()])

    out = {}
    for m in methods:
        out[m] = {
            "psnr": [],
            "ssim": []
        }

    for snr in snrs:
        entry = results[str(snr)]
        for m in methods:
            out[m]["psnr"].append(entry[m]["psnr"])
            out[m]["ssim"].append(entry[m]["ssim"])

    return snrs, out

# ====== LATEX TABLE ======
def generate_latex(channel_name, snrs, data):
    latex = []
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{%s Results}" % channel_name)
    latex.append("\\begin{tabular}{c|" + "c"*len(methods) + "}")
    latex.append("\\hline")
    latex.append("SNR & " + " & ".join(methods) + " \\\\ \\hline")

    # PSNR
    for i, snr in enumerate(snrs):
        row = [f"{snr}"]
        for m in methods:
            row.append(f"{data[m]['psnr'][i]:.2f}")
        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}\n")

    # SSIM table
    latex.append("\\begin{table}[h]")
    latex.append("\\centering")
    latex.append("\\caption{%s SSIM}" % channel_name)
    latex.append("\\begin{tabular}{c|" + "c"*len(methods) + "}")
    latex.append("\\hline")
    latex.append("SNR & " + " & ".join(methods) + " \\\\ \\hline")

    for i, snr in enumerate(snrs):
        row = [f"{snr}"]
        for m in methods:
            row.append(f"{data[m]['ssim'][i]:.3f}")
        latex.append(" & ".join(row) + " \\\\")

    latex.append("\\hline")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}\n")

    return "\n".join(latex)

# ====== PLOT ======
def plot_metric(snrs, data, metric, title, save_path):
    plt.figure()

    for m in methods:
        plt.plot(snrs, data[m][metric], marker='o', label=m)

    plt.xlabel("SNR (dB)")
    plt.ylabel(metric.upper())
    plt.title(title)
    plt.legend()
    plt.grid()

    plt.savefig(save_path)
    plt.close()

# ====== MAIN ======
output_tex = "results_tables.tex"

with open(output_tex, "w") as f_tex:

    for name, path in files.items():
        data = load_data(path)
        snrs, extracted = extract_results(data)

        # ===== LaTeX =====
        latex_str = generate_latex(name, snrs, extracted)
        f_tex.write(latex_str)

        # ===== Plot =====
        os.makedirs("plots", exist_ok=True)

        plot_metric(
            snrs, extracted,
            "psnr",
            f"{name} PSNR vs SNR",
            f"plots/{name}_psnr.png"
        )

        plot_metric(
            snrs, extracted,
            "ssim",
            f"{name} SSIM vs SNR",
            f"plots/{name}_ssim.png"
        )

print("Done! Generated LaTeX + plots/")