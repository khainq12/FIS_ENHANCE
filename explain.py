import torch
import matplotlib.pyplot as plt
import numpy as np


# =========================
# 1. Run model & extract info
# =========================
def run_explain(
    model,
    img,
    snr,
    channel_type,
    device="cuda",
    rician_k=4.0,
    rayleigh_equalize=False
):
    model.eval()

    with torch.no_grad():
        # đảm bảo shape (1,C,H,W)
        if img.dim() == 3:
            x = img.unsqueeze(0)
        else:
            x = img

        x = x.to(device)

        # set channel
        model.set_channel(
            channel_type=channel_type,
            snr=snr,
            rician_k=rician_k
        )

        # 🔥 FIX QUAN TRỌNG
        if hasattr(model, "channel"):
            model.channel.enable_rayleigh_equalization(rayleigh_equalize)

        # forward
        z_tx, x_hat, info = model(x, snr=snr, return_info=True)

    # ===== SAFE GET =====
    I = info.get("I", None)
    A = info.get("A", None)
    rule1 = info.get("rule1_id", None)
    rule2 = info.get("rule2_id", None)

    return x, x_hat, I, A, rule1, rule2


# =========================
# helper normalize
# =========================
def normalize_map(m):
    if m is None:
        return None

    m = m.detach().cpu()

    if m.dim() == 3:
        m = m.squeeze(0)

    m = m - m.min()
    if m.max() > 0:
        m = m / m.max()

    return m


# =========================
# helper rule hist
# =========================
def plot_rule_hist(ax, rule1, rule2):
    if rule1 is None or rule2 is None:
        ax.set_title("No Rules")
        return

    rules = torch.cat([rule1.view(-1), rule2.view(-1)]).cpu().numpy()

    bins = np.arange(-0.5, rules.max() + 1.5, 1)
    ax.hist(rules, bins=bins)
    ax.set_title("Rule Hist")


# =========================
# helper: denormalize CIFAR
# =========================
def denorm(x):
    return x * 0.5 + 0.5


# =========================
# 2. Save single figure
# =========================
def save_single(x, x_hat, I, A, rule1, rule2, title, save_path):
    fig, axs = plt.subplots(1, 5, figsize=(15, 3))

    I_vis = normalize_map(I[0]) if I is not None else None
    A_vis = normalize_map(A[0]) if A is not None else None

    axs[0].imshow(denorm(x[0]).permute(1, 2, 0).cpu().clamp(0, 1))
    axs[0].set_title("Input")

    if I_vis is not None:
        axs[1].imshow(I_vis, cmap='hot')
    axs[1].set_title("Importance")

    if A_vis is not None:
        axs[2].imshow(A_vis, cmap='viridis')
    axs[2].set_title("Power")

    axs[3].imshow(denorm(x_hat[0]).permute(1, 2, 0).cpu().clamp(0, 1))
    axs[3].set_title("Output")

    plot_rule_hist(axs[4], rule1, rule2)

    for ax in axs:
        ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# =========================
# 3. Save Figure 3
# =========================
def save_figure3(results_list, titles, save_path):
    n = len(results_list)
    fig, axs = plt.subplots(n, 5, figsize=(15, 5 * n))

    if n == 1:
        axs = [axs]

    for i, (res, title) in enumerate(zip(results_list, titles)):
        x, x_hat, I, A, rule1, rule2 = res

        I_vis = normalize_map(I[0]) if I is not None else None
        A_vis = normalize_map(A[0]) if A is not None else None

        axs[i][0].imshow(denorm(x[0]).permute(1, 2, 0).cpu().clamp(0, 1))
        axs[i][0].set_title("Input")

        if I_vis is not None:
            axs[i][1].imshow(I_vis, cmap='hot')
        axs[i][1].set_title("Importance")

        if A_vis is not None:
            axs[i][2].imshow(A_vis, cmap='viridis')
        axs[i][2].set_title("Power")

        axs[i][3].imshow(denorm(x_hat[0]).permute(1, 2, 0).cpu().clamp(0, 1))
        axs[i][3].set_title("Output")

        plot_rule_hist(axs[i][4], rule1, rule2)

        for j in range(5):
            axs[i][j].axis("off")

        axs[i][0].set_ylabel(title, fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()