"""Collect every completed run under <results>/runs into master tables and figures.

  <results>/master_runs.csv      one row per run (seed)
  <results>/master_table.csv/.md mean ± sd over seeds per tag (TEST columns only for --final runs)
  <results>/figs/*.png           privacy-utility, communication, fairness gaps, per-client AUROC
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from dpmmfl.cli import parser, config

J = lambda p: json.loads(Path(p).read_text())


def run_row(d):
    man = J(d / "manifest.json")
    cfg = yaml.safe_load((d / "config.yaml").read_text())
    v = J(d / "metrics_val.json")
    r = {"tag": man["tag"], "seed": man["seed"], "run_id": man["run_id"], "final": cfg["run"]["final"],
         "encoders": "+".join(cfg["features"]["encoders"]), "head": cfg["model"]["head"], "loss": cfg["model"]["loss"],
         "val_mean5": v["mean5_auroc"], "val_ci_lo": v.get("mean5_auroc_ci95", [np.nan] * 2)[0],
         "val_ci_hi": v.get("mean5_auroc_ci95", [np.nan] * 2)[1], "val_ece": v["ece"],
         "best_round": v["info"].get("best_round"), "rounds_run": v["info"].get("rounds_run"), "wall_sec": man.get("wall_sec")}
    if (d / "metrics_test.json").exists():
        t = J(d / "metrics_test.json")
        r.update(test_mean5=t["mean5_auroc"], test_ci_lo=t["mean5_auroc_ci95"][0], test_ci_hi=t["mean5_auroc_ci95"][1], test_ece=t["ece"])
    if (d / "privacy.json").exists():
        p = J(d / "privacy.json")
        r.update(eps_target=p.get("eps_target"), eps_prv=p.get("eps_spent_max_prv"), eps_rdp=p.get("eps_spent_max_rdp"),
                 eps_spread_prv=max(p["eps_spent_prv"]) - min(p["eps_spent_prv"]), steps=f"{p['actual_steps']}/{p['planned_steps']}")
    if (d / "comms.json").exists():
        c = J(d / "comms.json")
        r.update(comm=f"{c['mode_up']}/{c['mode_down']}", MB_total=c["MB_total"], saving_pct=c["saving_vs_fp32_pct"])
    if (d / "mia.json").exists():
        r["mia_auc"] = J(d / "mia.json")["mia_auc"]
    if (d / "fairness.json").exists():
        f = J(d / "fairness.json")
        for a, g in f["subgroups_val"].items():
            r[f"gap_{a}"], r[f"worst_{a}"] = g["max_gap"], g["worst_auroc"]
            r[f"eo_tpr_{a}"] = g["eo_tpr_gap"]
        if f.get("clients_localval"):
            r["gap_client"], r["worst_client"] = f["clients_localval"]["max_gap"], f["clients_localval"]["worst_auroc"]
        f2 = f["F2_group_thresholds"]
        r["F2_tpr_gap_global"], r["F2_tpr_gap_group"] = f2["global_thresholds"]["eo_tpr_gap"], f2["group_thresholds"]["eo_tpr_gap"]
    return r


def main():
    a = parser(__doc__).parse_args()
    cfg = config(a)
    root = Path(cfg["results_dir"])
    runs = [d for d in sorted((root / "runs").glob("*")) if (d / "DONE").exists()]
    if not runs:
        print(f"no completed runs under {root / 'runs'}"); sys.exit(1)
    df = pd.DataFrame([run_row(d) for d in runs])
    df.to_csv(root / "master_runs.csv", index=False)
    num = [c for c in df.columns if c not in ("tag", "seed", "run_id", "final", "encoders", "head", "loss", "comm", "steps") and pd.api.types.is_numeric_dtype(df[c])]
    g = df.groupby(["tag", "final"])
    tab = g[num].mean().round(4)
    for c in ("val_mean5", "test_mean5"):
        if c in df:
            tab[c + "_sd"] = g[c].std().round(4)
    tab.insert(0, "n_seeds", g.size())
    for c in ("encoders", "head", "loss", "comm", "steps"):
        if c in df:
            tab[c] = g[c].first()
    tab = tab.reset_index().sort_values(["final", "tag"], ascending=[False, True])
    tab.to_csv(root / "master_table.csv", index=False)
    show = [c for c in ["tag", "final", "n_seeds", "encoders", "head", "loss", "val_mean5", "val_mean5_sd", "test_mean5", "test_mean5_sd",
                        "eps_prv", "eps_rdp", "comm", "MB_total", "saving_pct", "gap_age", "gap_projection", "gap_client",
                        "F2_tpr_gap_global", "F2_tpr_gap_group", "mia_auc", "wall_sec"] if c in tab]
    try:
        md = tab[show].to_markdown(index=False, floatfmt=".4g")
    except ImportError:                                  # tabulate not installed
        md = tab[show].to_string(index=False)
    (root / "master_table.md").write_text(md)
    print(tab[show].to_string(index=False))

    figs = root / "figs"
    figs.mkdir(exist_ok=True)
    metric = "test_mean5" if "test_mean5" in tab and tab["test_mean5"].notna().any() else "val_mean5"
    made = []
    dp = tab[tab.tag.str.match(r"^E4_eps[\d.]+$") & tab.get("eps_prv", pd.Series(dtype=float)).notna()] if "eps_prv" in tab else tab.iloc[:0]
    if len(dp):
        dp = dp.sort_values("eps_prv")
        plt.figure(figsize=(5.5, 4))
        plt.errorbar(dp.eps_prv, dp[metric], yerr=dp.get(metric + "_sd", 0 * dp[metric]).fillna(0), marker="o", label="patient-level DP-FedAvg")
        for t, c in (("E3", "C1"), ("E2", "C2"), ("E0", "gray")):
            if t in set(tab.tag):
                plt.axhline(tab[tab.tag == t][metric].iloc[0], ls="--", color=c, label=t)
        plt.xscale("log"); plt.xlabel("ε spent (PRV, δ=1e-5)"); plt.ylabel(metric.replace("_", " ")); plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig(figs / "privacy_utility.png", dpi=140); plt.close(); made.append("privacy_utility.png")
    if "MB_total" in tab and tab.MB_total.notna().any():
        c = tab[tab.MB_total.notna()]
        plt.figure(figsize=(6, 4))
        plt.scatter(c.MB_total, c[metric])
        for _, r in c.iterrows():
            plt.annotate(r.tag, (r.MB_total, r[metric]), fontsize=6)
        plt.xlabel("measured MB transmitted (up + down)"); plt.ylabel(metric.replace("_", " ")); plt.tight_layout()
        plt.savefig(figs / "communication.png", dpi=140); plt.close(); made.append("communication.png")
    gaps = [c for c in ("gap_sex", "gap_age", "gap_projection", "gap_client") if c in tab]
    if gaps:
        tab.set_index("tag")[gaps].plot.bar(figsize=(max(6, 0.5 * len(tab)), 4))
        plt.ylabel("max - min group AUROC (VAL)"); plt.tight_layout()
        plt.savefig(figs / "fairness_gaps.png", dpi=140); plt.close(); made.append("fairness_gaps.png")
    hist = [d for d in runs if (d / "history.csv").exists() and "eps" in d.name and d.name.endswith("s0")]
    if hist:
        plt.figure(figsize=(6, 4))
        for d in hist[:8]:
            h = pd.read_csv(d / "history.csv")
            if "eps_rdp_max" in h:
                plt.plot(h["round"], h["eps_rdp_max"], label=d.name.split("__")[0])
        plt.xlabel("round"); plt.ylabel("max client ε (RDP, running)"); plt.legend(fontsize=6); plt.tight_layout()
        plt.savefig(figs / "eps_vs_round.png", dpi=140); plt.close(); made.append("eps_vs_round.png")
    print(f"\n{len(df)} runs, {len(tab)} configs -> {root / 'master_table.csv'}\nfigures: {', '.join(made)}")


if __name__ == "__main__":
    main()
