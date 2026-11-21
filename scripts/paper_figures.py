"""Paper / report figures and tables from the v2.1 results. Reads results only (and, for Grad-CAM, the saved demo model).

  python scripts/paper_figures.py [--out paper] [--no-gradcam]

Writes paper/figures/fig*.{pdf,png} and paper/tables/*.{tex,csv}. TEST numbers come from the stored final runs; nothing is re-evaluated.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dpmmfl.constants import COMP5, LABELS  # noqa: E402

FINAL = ROOT / "results/v2.1-final"
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRAY,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2, "figure.dpi": 150, "savefig.bbox": "tight",
                     "pdf.fonttype": 42})


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print("wrote", name)


def runs():
    r = pd.read_csv(FINAL / "master_runs.csv")
    r = r[r.final.astype(bool)].copy()
    r["dir"] = [FINAL / "runs" / i for i in r.run_id]
    return r


def ms(r, tag, col):
    v = r.loc[r.tag == tag, col]
    return v.mean(), v.std()


# ---------------------------------------------------------------- 1. architecture
def fig_architecture(out):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(-1, 102), ax.set_ylim(0, 50), ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f4f3ef", ec=GRAY):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h - 1.4, title, ha="center", va="top", fontsize=6.8, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 4.8, body.replace("|", "\n"), ha="center", va="top", fontsize=5.8, color=INK2, linespacing=1.35)

    def arrow(p, q, text=None, dx=0, dy=1.2):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=8, lw=0.9, color=INK2))
        if text:
            ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, text, ha="center", fontsize=6, color=INK2, style="italic")

    box(0, 29, 16, 17, "CheXpert (small)", "191,027 frontal images|64,534 patients|14 findings|+ age, sex, projection")
    box(20, 29, 19, 17, "Frozen encoders", "TorchXRayVision|DenseNet-121 ×3|(nih, mimic_ch, pc)|no CheXpert weights|"
        "3 × 1024-d, cached once")
    box(0, 6, 16, 19, "Non-IID split", "patient-disjoint|FL pool / VAL / TEST|5 clients, Dirichlet|α = 0.5|on label mix")
    box(20, 6, 19, 19, "Client k (×5)", "FiLM head on own data|per-patient gradient|clip to C, add N(0, σ²C²)|σₖ set so every client|"
        "spends the same ε", fc="#eaf2fc", ec=BLUE)
    box(44, 6, 17, 19, "Compression", "int8 per block|(or top-k + error|feedback), applied|after the noise:|ε unchanged")
    box(66, 6, 14, 19, "Server", "FedAvg / FedAvgM|weighted by|client size||fp16 broadcast")
    box(44, 29, 36, 17, "Global FusionHead (1.09M parameters)", "per encoder: LayerNorm → Linear(1024, 256)|"
        "concatenate, FiLM on 9 metadata features|BCE with pos_weight")
    box(85, 20, 16, 26, "Post-processing", "temperature|calibration (VAL)||per-group|thresholds (F2)", fc="#e8f6f0", ec=AQUA)
    box(85, 6, 16, 10, "Audit", "AUROC + CIs|fairness, MIA|ε: PRV + RDP")

    arrow((16, 37.5), (20, 37.5))
    arrow((8, 29), (8, 25))
    arrow((16, 15.5), (20, 15.5))
    arrow((29.5, 29), (29.5, 25), "features", dx=5, dy=-0.6)
    arrow((39, 15.5), (44, 15.5), "Δθₖ")
    arrow((61, 15.5), (66, 15.5))
    arrow((73, 25), (73, 29), "θ", dx=2, dy=-0.6)
    arrow((80, 37.5), (85, 37.5))
    arrow((93, 20), (93, 16))
    ax.plot([73, 73, 29.5], [6, 2.6, 2.6], color=INK2, lw=0.9)
    arrow((29.5, 2.6), (29.5, 6))
    ax.text(51, 0.4, "next round: global θ sent to clients (fp16)", ha="center", fontsize=6, color=INK2, style="italic")
    ax.text(0, 48.6, "Patient-level DP boundary: noise is added on each client; everything after it is post-processing.", fontsize=6.3, color=BLUE)
    save(fig, out, "fig1_architecture")


# ---------------------------------------------------------------- 2. privacy-utility
def fig_privacy_utility(r, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    eps = [1, 3, 8]
    for ax, col, name in zip(axes, ("val_mean5", "test_mean5"), ("VAL (6,453 patients)", "TEST (202 images)")):
        m = [ms(r, f"E4_eps{e}", col) for e in eps]
        ax.errorbar(eps, [a for a, _ in m], [b for _, b in m], color=BLUE, marker="o", ms=5, capsize=3, label="patient-level DP-FL")
        im = ms(r, "E5_eps8", col)
        ax.errorbar([8.6], [im[0]], [im[1]], color=ORANGE, marker="s", ms=5, capsize=3, lw=0, elinewidth=1.5, label="image-level DP, ε=8")
        for tag, ls, lab in (("E2", "-", "central (FiLM)"), ("E3", "--", "FedAvg, no DP")):
            v = ms(r, tag, col)[0]
            ax.axhline(v, color=GRAY, ls=ls, lw=1.2)
            ax.text(1.0, v, f" {lab} {v:.3f}", va="bottom", fontsize=6.5, color=INK2)
        ax.set_xscale("log"), ax.set_xticks(eps, ["1", "3", "8"]), ax.minorticks_off()
        ax.set_xlabel("privacy budget ε (δ = 1e-5)"), ax.set_ylabel("mean AUROC, 5 findings"), ax.set_title(name, loc="left")
    axes[0].legend(loc="lower right", frameon=False)
    fig.tight_layout()
    save(fig, out, "fig2_privacy_utility")


# ---------------------------------------------------------------- 3. scale ladder
def fig_scale(out):
    s = pd.read_csv(ROOT / "results/scale_comparison.csv")
    order = ["small (2k pts)", "medium (20k pts)", "large (64k pts, v2)", "large (64k pts, v2.1)"]
    xt = ["small\n2k pool patients", "medium\n20k", "large v2\n58k, 1 encoder", "large v2.1\n58k, 3 encoders"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for cfg, c, lab in (("central, FiLM", GRAY, "central"), ("FedAvg", BLUE, "FedAvg"), ("DP eps=3", ORANGE, "DP ε=3"), ("DP eps=1", AQUA, "DP ε=1")):
        d = s[s.config == cfg].set_index("scale").reindex(order)
        ax.errorbar(range(4), d.val, d.val_sd.fillna(0), color=c, marker="o", ms=4, capsize=2, label=lab)
    ax.set_xticks(range(4), xt), ax.set_ylabel("VAL mean AUROC"), ax.set_title("Accuracy grows with data", loc="left")
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax = axes[1]
    cost = [s[(s.config == "FedAvg") & (s.scale == k)].val.iloc[0] - s[(s.config == "DP eps=3") & (s.scale == k)].val.iloc[0] for k in order]
    b = ax.bar(range(4), cost, color=ORANGE, width=0.6, edgecolor="white", linewidth=1)
    ax.bar_label(b, [f"{c:.3f}" for c in cost], fontsize=6.5, color=INK, padding=2)
    ax.set_xticks(range(4), ["small", "medium", "large\nv2", "large\nv2.1"]), ax.set_ylabel("VAL AUROC lost (FedAvg − DP)")
    ax.set_title("Cost of DP at ε=3 shrinks", loc="left"), ax.set_ylim(0, max(cost) * 1.2)
    fig.tight_layout()
    save(fig, out, "fig3_scale_ladder")


# ---------------------------------------------------------------- 4. communication
def fig_comm(r, out):
    rows = [("E3", "fp32", "FedAvg"), ("E8_upint8_dnfp16", "int8 / fp16", "FedAvg"), ("E4_eps3", "fp32", "DP ε=3"),
            ("E8_eps3_upint8_dnfp16", "int8 / fp16", "DP ε=3"), ("E8_eps3_uptopk_dnfp16", "top-k / fp16", "DP ε=3")]
    lab = [f"{g}\n{c}" for _, c, g in rows]
    mb = [ms(r, t, "MB_total")[0] for t, _, _ in rows]
    au = [ms(r, t, "val_mean5") for t, _, _ in rows]
    cols = [GRAY if c == "fp32" else BLUE if "int8" in c else ORANGE for _, c, _ in rows]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    b = axes[0].barh(range(5), mb, color=cols, height=0.6, edgecolor="white")
    axes[0].bar_label(b, [f"{m:.0f} MB" + ("" if c == "fp32" else f"  (−{100 * (1 - m / mb[0 if i < 2 else 2]):.0f}%)")
                          for i, (m, (_, c, _)) in enumerate(zip(mb, rows))], fontsize=6.5, padding=2)
    axes[0].set_yticks(range(5), lab), axes[0].invert_yaxis(), axes[0].set_xlim(0, max(mb) * 1.45)
    axes[0].set_xlabel("total traffic per training run (up + down)"), axes[0].set_title("Bytes on the wire", loc="left")
    for i, ((m, sd), c) in enumerate(zip(au, cols)):
        axes[1].errorbar(m, i, xerr=sd, color=c, marker="o", ms=5, capsize=3)
        axes[1].text(m, i - 0.3, f"{m:.3f}", ha="center", fontsize=6.5, color=INK)
    axes[1].set_yticks(range(5), lab), axes[1].invert_yaxis(), axes[1].set_xlabel("VAL mean AUROC"), axes[1].set_title("Accuracy", loc="left")
    fig.tight_layout()
    save(fig, out, "fig4_communication")


# ---------------------------------------------------------------- 5. training curves + epsilon
def fig_curves(r, out):
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for tag, c, lab in (("E3", GRAY, "FedAvg"), ("E4_eps8", BLUE, "DP ε=8"), ("E4_eps3", ORANGE, "DP ε=3"), ("E4_eps1", AQUA, "DP ε=1")):
        h = pd.read_csv(r[(r.tag == tag) & (r.seed == 0)].dir.iloc[0] / "history.csv")
        axes[0].plot(h["round"], h.val_mean5_auroc, color=c, label=lab)
        if tag != "E3":
            axes[1].plot(h["round"], h.eps_rdp_max, color=c, label=lab)
            axes[1].axhline(int(tag[-1]), color=c, lw=0.8, ls=":")
            prv = max(json.loads((r[(r.tag == tag) & (r.seed == 0)].dir.iloc[0] / "privacy.json").read_text())["eps_spent_prv"])
            axes[1].plot(h["round"].iloc[-1], prv, "D", color=c, ms=5, mec="white")
    axes[0].set_xlabel("communication round"), axes[0].set_ylabel("VAL mean AUROC"), axes[0].set_title("Training (seed 0)", loc="left")
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].set_xlabel("communication round"), axes[1].set_ylabel("ε spent (RDP bound, worst client)")
    axes[1].set_title("ε spent: line = RDP bound, ◆ = final PRV, dotted = target", loc="left"), axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, out, "fig5_training_and_epsilon")


# ---------------------------------------------------------------- 6. fairness
def fig_fairness(r, out):
    runs3 = (("E2", GRAY, "central"), ("E3", BLUE, "FedAvg"), ("E4_eps3", ORANGE, "DP ε=3"))
    groups = [("age", g) for g in ("<40", "40-60", "60-80", ">=80")] + [("projection", g) for g in ("PA", "AP")] + [("sex", g) for g in ("Female", "Male")]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axes[0]
    for j, (tag, c, lab) in enumerate(runs3):
        f = json.loads((r[(r.tag == tag) & (r.seed == 0)].dir.iloc[0] / "fairness.json").read_text())["subgroups_val"]
        for i, (a, g) in enumerate(groups):
            d = f[a]["groups"][g]
            ax.errorbar(i + (j - 1) * 0.22, d["auroc"], [[d["auroc"] - d["ci95"][0]], [d["ci95"][1] - d["auroc"]]], color=c, marker="o",
                        ms=4, capsize=2, lw=0, elinewidth=1.2, label=lab if i == 0 else None)
    ax.set_xticks(range(len(groups)), [f"{g}" for _, g in groups]), ax.set_ylabel("VAL mean AUROC (95% CI)")
    for x, t in ((1.5, "age"), (4.5, "projection"), (6.5, "sex")):
        ax.text(x, ax.get_ylim()[0], t, ha="center", va="bottom", fontsize=6.5, color=INK2, style="italic")
    for x in (3.5, 5.5):
        ax.axvline(x, color="#d6d5cf", lw=0.8)
    ax.set_title("Subgroup AUROC (seed 0)", loc="left"), ax.legend(frameon=False, ncol=3, loc="upper right")
    ax = axes[1]
    tags = (("E3", "FedAvg"), ("E4_eps3", "DP ε=3"))
    for k, (col, c, lab) in enumerate((("F2_tpr_gap_global", GRAY, "one global threshold"), ("F2_tpr_gap_group", AQUA, "per-group thresholds (F2)"))):
        v = [ms(r, t, col) for t, _ in tags]
        b = ax.bar(np.arange(2) + (k - 0.5) * 0.36, [a for a, _ in v], 0.34, yerr=[s for _, s in v], color=c, capsize=2,
                   error_kw={"ecolor": INK2, "lw": 1}, edgecolor="white", label=lab)
        ax.bar_label(b, [f"{a:.3f}" for a, _ in v], fontsize=6.5, padding=2)
    ax.set_xticks(range(2), [n for _, n in tags]), ax.set_ylabel("sensitivity gap |TPR(AP) − TPR(PA)|")
    ax.set_title("Mitigation F2 (held-out VAL)", loc="left"), ax.legend(frameon=False, loc="upper right"), ax.set_ylim(0, 0.3)
    fig.tight_layout()
    save(fig, out, "fig6_fairness")


# ---------------------------------------------------------------- 7. membership inference
def fig_mia(r, out):
    s = pd.read_csv(ROOT / "results/mia_demo/summary.csv")
    arms = ["no DP", "DP eps=8", "DP eps=3", "DP eps=1"]
    g = s.groupby("arm").auc.agg(["mean", "std"]).reindex(arms)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.1, 1, 1]})
    b = axes[0].bar(range(4), g["mean"], yerr=g["std"], color=[GRAY, BLUE, BLUE, BLUE], width=0.6, capsize=2, error_kw={"ecolor": INK2, "lw": 1}, edgecolor="white")
    axes[0].bar_label(b, [f"{v:.2f}" for v in g["mean"]], fontsize=6.5, padding=2)
    axes[0].axhline(0.5, color=INK2, ls=":", lw=1), axes[0].text(3.45, 0.465, "chance", ha="right", fontsize=6, color=INK2)
    axes[0].set_xticks(range(4), ["no DP", "ε=8", "ε=3", "ε=1"]), axes[0].set_ylim(0.4, 1.0)
    axes[0].set_ylabel("attack AUC (5 seeds)"), axes[0].set_title("Stress test, 300 patients", loc="left")
    l = pd.read_csv(ROOT / "results/mia_demo/losses_seed0.csv")
    for ax, arm, t in ((axes[1], "no DP", "no DP: members separable"), (axes[2], "DP eps=3", "DP ε=3: indistinguishable")):
        d = l[l.arm == arm]
        bins = np.linspace(0, np.quantile(d.loss, 0.98), 30)
        ax.hist(d[d.member].loss, bins, color=ORANGE, alpha=0.55, label="training patients")
        ax.hist(d[~d.member].loss, bins, color=BLUE, histtype="step", lw=1.5, label="held-out patients")
        ax.set_xlabel("per-patient loss"), ax.set_title(t, loc="left"), ax.set_yticks([])
    axes[1].legend(frameon=False)
    fig.tight_layout()
    save(fig, out, "fig7_membership_inference")


# ---------------------------------------------------------------- 8. per-label heatmap
def fig_per_label(r, out):
    tags = [("E0", "linear probe"), ("E2", "central"), ("E3", "FedAvg"), ("E4_eps8", "DP ε=8"), ("E4_eps3", "DP ε=3"), ("E4_eps1", "DP ε=1")]
    M = np.array([[np.nanmean([json.loads((d / "metrics_val.json").read_text())["auroc"].get(l) or np.nan for d in r[r.tag == t].dir])
                   for t, _ in tags] for l in LABELS], dtype=float)
    keep = ~np.isnan(M).all(1)
    labs = [l + (" *" if l in COMP5 else "") for l in np.array(LABELS)[keep]]
    M = M[keep]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(M, cmap="Blues", vmin=0.6, vmax=0.95, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6, color="white" if M[i, j] > 0.82 else INK)
    ax.set_xticks(range(len(tags)), [n for _, n in tags], rotation=30, ha="right"), ax.set_yticks(range(len(labs)), labs)
    ax.grid(False), ax.set_title("VAL AUROC per finding (3 seeds)  * = competition 5", loc="left")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    save(fig, out, "fig8_per_label_auroc")


# ---------------------------------------------------------------- 9. non-IID partition
def fig_partition(out):
    m = pd.read_parquet(ROOT / "cache/meta_train.parquet", columns=["patient_id", "split", "client_id"] + [f"{p}_{l}" for l in COMP5 for p in "TM"])
    m = m[m.split == "pool"]
    pts = m.groupby("client_id").patient_id.nunique()
    prev = pd.DataFrame({l: (m[f"T_{l}"] * (m[f"M_{l}"] > 0)).groupby(m.client_id).sum() / (m[f"M_{l}"] > 0).groupby(m.client_id).sum()
                         for l in COMP5})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1, 2]})
    b = axes[0].barh(range(len(pts)), pts.values, color=BLUE, height=0.6)
    axes[0].bar_label(b, [f"{v:,}" for v in pts.values], fontsize=6.5, padding=2)
    axes[0].set_yticks(range(len(pts)), [f"client {int(c) + 1}" for c in pts.index]), axes[0].invert_yaxis()
    axes[0].set_xlim(0, pts.max() * 1.3), axes[0].set_xlabel("patients"), axes[0].set_title("Client sizes", loc="left")
    im = axes[1].imshow(prev.values, cmap="Blues", vmin=0, vmax=prev.values.max(), aspect="auto")
    for i in range(prev.shape[0]):
        for j in range(prev.shape[1]):
            v = prev.values[i, j]
            axes[1].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if v > prev.values.max() * 0.6 else INK)
    axes[1].set_xticks(range(5), COMP5, rotation=15), axes[1].set_yticks(range(len(pts)), [f"client {int(c) + 1}" for c in pts.index])
    axes[1].grid(False), axes[1].set_title("Label prevalence per client (Dirichlet α = 0.5)", loc="left")
    fig.colorbar(im, ax=axes[1], fraction=0.03, pad=0.02)
    fig.tight_layout()
    save(fig, out, "fig9_noniid_partition")


# ---------------------------------------------------------------- 10. Grad-CAM
def fig_gradcam(out):
    import torch
    from PIL import Image
    from dpmmfl.serve import Predictor
    model = ROOT / "models/v2.1-large/E4_eps3_s0"
    s = pd.read_csv(ROOT / "demo/demo_samples.csv")
    s = s[(s.split == "val") & s.finding.isin(["Cardiomegaly", "Edema", "Pleural Effusion", "Atelectasis"])].groupby("finding").head(1)
    P = Predictor(model)
    fig, axes = plt.subplots(2, len(s), figsize=(7.0, 3.8))
    for j, (_, row) in enumerate(s.iterrows()):
        gray = np.array(Image.open(ROOT / row.image).convert("L"))
        meta = dict(age=float(row.age) if pd.notna(row.age) else None, sex=row.sex, projection=row.projection)
        df, x = P.predict(gray, **meta)
        cam, _ = P.cam(x, row.finding, **meta)
        img = x[0, 0].cpu().numpy()
        for ax in axes[:, j]:
            ax.imshow(img, cmap="gray"), ax.axis("off")
        axes[1, j].imshow(cam, cmap="inferno", alpha=0.45)
        p = float(df.set_index("label").prob[row.finding])
        axes[0, j].set_title(f"{row.finding}\n{row.sex}, {int(row.age)}, {row.projection}", fontsize=7)
        axes[1, j].set_title(f"P = {p:.2f}", fontsize=7)
        torch.cuda.empty_cache()
    fig.suptitle("Grad-CAM, DP model (ε = 3), held-out VAL patients", fontsize=8, x=0.01, ha="left")
    fig.tight_layout()
    save(fig, out, "fig10_gradcam")


# ---------------------------------------------------------------- 11. LiRA
def fig_lira(out, tables_out):
    d = ROOT / "results/mia_lira"
    if not (d / "stress/summary.csv").exists():
        print("skip fig11 (run scripts/mia_lira.py stress)")
        return
    S, R = pd.read_csv(d / "stress/summary.csv"), pd.read_csv(d / "stress/roc.csv")
    arms = list(dict.fromkeys(S.arm))
    style = {"no DP (overfit)": (GRAY, "-"), "no DP, early stopped": (GRAY, "--"), "DP eps=8": (BLUE, "-"), "DP eps=3": (ORANGE, "-"),
             "DP eps=1": (AQUA, "-")}
    lab = lambda a: a.replace("eps", "ε")  # noqa: E731
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"width_ratios": [1.05, 1.35, 0.8]})
    ax = axes[0]
    for a in arms:
        r = R[(R.arm == a) & (R.attack == "LiRA online") & (R.fpr > 0)]
        c, ls = style.get(a, (INK2, "-"))
        ax.plot(r.fpr, r.tpr, color=c, ls=ls, lw=1.5, label=lab(a))
    f = np.geomspace(1e-3, 1, 100)
    ax.plot(f, f, color=INK2, ls=":", lw=1)
    ax.set_xscale("log"), ax.set_yscale("log"), ax.set_xlim(1e-3, 1), ax.set_ylim(1e-3, 1)
    ax.set_xlabel("false-positive rate"), ax.set_ylabel("true-positive rate"), ax.set_title("LiRA (online), ROC", loc="left")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax = axes[1]
    shades = {"loss threshold": "#b7d3f2", "LiRA offline": "#6aa5e6", "LiRA online": "#1d5fae"}
    for j, (att, c) in enumerate(shades.items()):
        v = [S[(S.arm == a) & (S.attack == att)]["tpr@0.01"].iloc[0] * 100 for a in arms]
        ax.bar(np.arange(len(arms)) + (j - 1) * 0.27, v, 0.25, color=c, edgecolor="white", label=att)
    ax.axhline(1, color=INK2, ls=":", lw=1), ax.text(len(arms) - 0.55, 2.2, "chance (1%)", ha="right", fontsize=6, color=INK2)
    short = {"no DP (overfit)": "no DP\noverfit", "no DP, early stopped": "no DP\nearly stop"}
    ax.set_xticks(range(len(arms)), [short.get(a, lab(a).replace("DP ", "DP\n")) for a in arms], fontsize=6)
    ax.set_ylabel("training patients found (%) at 1% FPR"), ax.set_title("Attack strength", loc="left"), ax.legend(frameon=False, fontsize=6)
    ax = axes[2]
    dp = S[S.eps_claimed.notna()].groupby("arm", sort=False).agg(claimed=("eps_claimed", "first"), lb=("eps_lower_bound", "max"))
    x = np.arange(len(dp))
    ax.bar(x - 0.18, dp.claimed, 0.34, color=GRAY, edgecolor="white", label="guaranteed (accountant)")
    b = ax.bar(x + 0.18, dp.lb, 0.34, color=ORANGE, edgecolor="white", label="shown by best attack")
    ax.bar_label(b, [f"{v:.2f}" for v in dp.lb], fontsize=6, padding=2)
    ax.set_xticks(x, [lab(a).replace("DP ", "") for a in dp.index]), ax.set_ylabel("ε"), ax.set_title("DP audit", loc="left")
    ax.legend(frameon=False, fontsize=6, loc="upper right"), ax.set_ylim(0, dp.claimed.max() * 1.45)
    fig.tight_layout()
    save(fig, out, "fig11_lira_attack")
    f3 = lambda v: "--" if pd.isna(v) else f"{v:.3f}"  # noqa: E731
    lines = [r"\begin{table}[t]\centering\small", r"\caption{Membership inference. Stress test: small head on 600 candidate patients, "
             r"16 shadow models per arm, leave-one-out. TPR at 1\% and 0.1\% FPR; $\hat\varepsilon_{\mathrm{lb}}$ = empirical lower bound "
             r"(Clopper--Pearson, 95\%).}", r"\label{tab:mia}", r"\begin{tabular}{llcccc}", r"\toprule",
             r"Training & Attack & AUC & TPR@1\% & TPR@0.1\% & $\hat\varepsilon_{\mathrm{lb}}$ \\", r"\midrule"]
    for _, r in S.iterrows():
        lines.append(f"{r.arm} & {r.attack} & {f3(r.auc)} & {f3(r['tpr@0.01'])} & {f3(r['tpr@0.001'])} & {f3(r.eps_lower_bound)} \\\\")
    if (d / "real/summary.csv").exists():
        lines += [r"\midrule", r"\multicolumn{6}{l}{\emph{Full-scale models (2{,}000 members vs 2{,}000 VAL patients, 8 reference models)}} \\"]
        names = {"E2": "Central (E2)", "E3": "FedAvg (E3)", "E4_eps8": "DP-FL eps=8", "E4_eps3": "DP-FL eps=3", "E4_eps1": "DP-FL eps=1"}
        for _, r in pd.read_csv(d / "real/summary.csv").iterrows():
            lines.append(f"{names.get(r.target, r.target.replace('_', ' '))} &{r.attack} & {f3(r.auc)} & {f3(r['tpr@0.01'])} & {f3(r['tpr@0.001'])} & {f3(r.eps_lower_bound)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (tables_out / "mia_results.tex").write_text(latex(lines).replace("eps=", "ε=").replace("ε=", r"$\varepsilon$="), encoding="utf8")


# ---------------------------------------------------------------- tables
def latex(lines):
    """pdflatex-safe: unicode ε and en dash replaced by LaTeX commands."""
    return "\n".join(lines).replace("ε", r"$\varepsilon$").replace("–", "--") + "\n"


def tables(r, out):
    rows = [("E0", "Linear probe (logistic regression)"), ("E1", "Central, image only"), ("E2", "Central, FiLM (image + metadata)"),
            ("E3", "FedAvg, 5 non-IID clients"), ("E4_eps8", "Patient-level DP-FL, ε = 8"), ("E4_eps3", "Patient-level DP-FL, ε = 3"),
            ("E4_eps1", "Patient-level DP-FL, ε = 1"), ("E5_eps8", "Image-level DP-FL, ε = 8"), ("E7_eps3_F1", "DP ε = 3 + group reweighting"),
            ("E8_upint8_dnfp16", "FedAvg + int8/fp16"), ("E8_eps3_upint8_dnfp16", "DP ε = 3 + int8/fp16"), ("E8_eps3_uptopk_dnfp16", "DP ε = 3 + top-k/fp16")]
    recs = []
    for t, name in rows:
        g = r[r.tag == t]
        recs.append(dict(run=name, tag=t, val=g.val_mean5.mean(), val_sd=g.val_mean5.std(), test=g.test_mean5.mean(), test_sd=g.test_mean5.std(),
                         test_ci=f"[{g.test_ci_lo.mean():.3f}, {g.test_ci_hi.mean():.3f}]", eps_prv=g.eps_prv.max(), eps_rdp=g.eps_rdp.max(),
                         MB=g.MB_total.mean(), mia=g.mia_auc.mean(), gap_age=g.gap_age.mean(), gap_proj=g.gap_projection.mean()))
    t = pd.DataFrame(recs)
    t.to_csv(out / "main_results.csv", index=False)
    f = lambda v, d=3: "–" if pd.isna(v) else f"{v:.{d}f}"  # noqa: E731
    lines = [r"\begin{table}[t]\centering\small", r"\caption{Mean AUROC over the five CheXpert competition findings (mean $\pm$ sd, 3 seeds). "
             r"TEST = 202 official validation images; CI = patient bootstrap, averaged over seeds. $\varepsilon$ at $\delta=10^{-5}$, worst client.}",
             r"\label{tab:main}", r"\begin{tabular}{lccccc}", r"\toprule",
             r"Run & VAL & TEST & TEST 95\% CI & $\varepsilon_{\mathrm{PRV}}$ & $\varepsilon_{\mathrm{RDP}}$ \\", r"\midrule"]
    for _, x in t.iterrows():
        lines.append(f"{x.run} & {f(x.val)} $\\pm$ {f(x.val_sd)} & {f(x.test)} $\\pm$ {f(x.test_sd)} & {x.test_ci} & {f(x.eps_prv, 2)} & {f(x.eps_rdp, 2)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "main_results.tex").write_text(latex(lines), encoding="utf8")
    lines = [r"\begin{table}[t]\centering\small", r"\caption{Communication, fairness and membership inference (means over 3 seeds; gaps on VAL).}",
             r"\label{tab:secondary}", r"\begin{tabular}{lcccc}", r"\toprule", r"Run & MB / run & age gap & projection gap & MIA AUC \\", r"\midrule"]
    for _, x in t.iterrows():
        lines.append(f"{x.run} & {f(x.MB, 0)} & {f(x.gap_age)} & {f(x.gap_proj)} & {f(x.mia)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "secondary_results.tex").write_text(latex(lines), encoding="utf8")
    print("wrote tables")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="paper")
    ap.add_argument("--no-gradcam", action="store_true")
    a = ap.parse_args()
    fo, to = ROOT / a.out / "figures", ROOT / a.out / "tables"
    fo.mkdir(parents=True, exist_ok=True), to.mkdir(parents=True, exist_ok=True)
    r = runs()
    fig_architecture(fo)
    fig_privacy_utility(r, fo)
    fig_scale(fo)
    fig_comm(r, fo)
    fig_curves(r, fo)
    fig_fairness(r, fo)
    fig_mia(r, fo)
    fig_per_label(r, fo)
    fig_partition(fo)
    if not a.no_gradcam:
        fig_gradcam(fo)
    fig_lira(fo, to)
    tables(r, to)


if __name__ == "__main__":
    main()
