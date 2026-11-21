"""Small vs medium vs large: results/scale_comparison.{csv,md} + figs.

small / medium: results/scale_<name>/master_runs.csv (v2.1 code, nih encoder, 15 rounds x 50 steps).
large (v2): the v2 full-data runs (results/v2-large/summary.csv;
64k patients, 30 rounds x 50 steps, v2 code) - protocol differs, labelled in the table.
large (v2.1): results/v2.1-final/master_runs.csv, added as a column once the overnight run has been aggregated.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parents[1] / "results"
TAGS = [("E1", "central, image-only", "E1", "E1"), ("E2", "central, FiLM", "E2", "E2"), ("E3", "FedAvg", "E3", "E3"),
        ("E4_eps8", "DP eps=8", "E4_eps8", "E4_eps8"), ("E4_eps3", "DP eps=3", "E4_eps3", "E4_eps3"), ("E4_eps1", "DP eps=1", "E4_eps1", "E4_eps1"),
        ("E5_eps8", "image-level DP eps=8", "E5_eps8", "E5_eps8"),
        ("E8_int8", "FedAvg int8 up / fp16 down", "E8_upint8_dnfp16", "E3_upint8_dnfp16")]


def small_med(folder):
    f = R / folder / "master_runs.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    d = d[d.final.astype(str) == "True"]
    out = {}
    for tag, g in d.groupby("tag"):
        out[tag] = dict(n=len(g), val=g.val_mean5.mean(), val_sd=g.val_mean5.std(), test=g.test_mean5.mean(), test_sd=g.test_mean5.std(),
                        eps=g.get("eps_prv", pd.Series([np.nan])).max(), mb=g.get("MB_total", pd.Series([np.nan])).mean())
    return out


def large():
    f = R / "v2-large" / "summary.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f).set_index("run")
    return {k: dict(n=int(r.n), val=r.val, val_sd=np.nan, test=r.test, test_sd=r.test_std, eps=r.eps_max, mb=np.nan) for k, r in d.iterrows()}


def main():
    src = {"small (2k pts)": small_med("scale_small"), "medium (20k pts)": small_med("scale_medium"), "large (64k pts, v2)": large()}
    if small_med("v2.1-final"):
        src["large (64k pts, v2.1)"] = small_med("v2.1-final")
    src = {k: v for k, v in src.items() if v}
    rows = []
    for _key, label, tag_new, tag_old in TAGS:
        for scale, d in src.items():
            t = tag_old if scale.endswith("v2)") else tag_new
            hit = d.get(t)
            if hit:
                rows.append(dict(config=label, scale=scale, **hit))
    df = pd.DataFrame(rows)
    df.to_csv(R / "scale_comparison.csv", index=False)
    piv = {m: df.pivot(index="config", columns="scale", values=m).reindex([t[1] for t in TAGS]) for m in ("val", "test")}
    sd = {m: df.pivot(index="config", columns="scale", values=m + "_sd").reindex([t[1] for t in TAGS]) for m in ("val", "test")}
    cols = list(src)
    lines = ["| config | " + " | ".join(f"{c} VAL" for c in cols) + " | " + " | ".join(f"{c} TEST" for c in cols) + " |", "|---|" + "---|" * (2 * len(cols))]
    fmt = lambda m, c, k: ("-" if pd.isna(piv[m].loc[c, k]) else f"{piv[m].loc[c, k]:.3f}" + ("" if pd.isna(sd[m].loc[c, k]) else f"±{sd[m].loc[c, k]:.3f}"))
    for c in piv["val"].index:
        lines.append(f"| {c} | " + " | ".join(fmt("val", c, k) for k in cols) + " | " + " | ".join(fmt("test", c, k) for k in cols) + " |")
    n = df.groupby("scale").n.max().to_dict()
    lines += ["", "seeds per scale (max over configs): " + ", ".join(f"{k}: {int(v)}" for k, v in n.items())]
    if "E3" in set(df.config) or True:
        gap = []
        for k in cols:
            a, b = piv["val"].loc["FedAvg", k], piv["val"].loc["DP eps=3", k]
            gap.append(f"{k}: {a - b:+.3f}" if not (pd.isna(a) or pd.isna(b)) else f"{k}: -")
        lines.append("DP(eps=3) VAL cost vs FedAvg: " + ", ".join(gap))
    (R / "scale_comparison.md").write_text("\n".join(lines))
    print("\n".join(lines))
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, m in zip(axs, ("val", "test")):
        piv[m].plot.bar(ax=ax, yerr=sd[m].fillna(0), capsize=2, width=0.8)
        ax.set_ylim(0.6, 0.86 if m == "test" else 0.78); ax.set_ylabel(f"{m.upper()} mean-5 AUROC"); ax.set_xlabel("")
        ax.tick_params(axis="x", labelrotation=30, labelsize=8); ax.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(R / "scale_comparison.png", dpi=140)
    print("saved", R / "scale_comparison.png")


if __name__ == "__main__":
    sys.exit(main())
