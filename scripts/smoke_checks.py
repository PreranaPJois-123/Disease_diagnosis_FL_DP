"""Assertions over the smoke runs (results/smoke). Exits non-zero if any check fails."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "results" / "smoke"
J = lambda p: json.loads(Path(p).read_text())
FAILS = []


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def run(tag):
    hits = [d for d in (ROOT / "runs").glob(f"{tag}__*") if (d / "DONE").exists()]
    if not hits:
        check(f"{tag} completed", False); return None
    return max(hits, key=lambda d: d.stat().st_mtime)


runs = {t: run(t) for t in ("E0", "E1", "E2_film_asl", "E3", "E4_eps3", "E8_upint8_dnfp16", "E8_eps3_uptopk_dnfp16", "E7_eps3_F1")}
rows = []
for t, d in runs.items():
    if d is None:
        continue
    v = J(d / "metrics_val.json")
    r = {"run": t, "val_mean5": round(v["mean5_auroc"], 4), "wall_s": J(d / "manifest.json")["wall_sec"]}
    if (d / "privacy.json").exists():
        p = J(d / "privacy.json")
        r.update(eps_prv=round(p["eps_spent_max_prv"], 3), eps_rdp=round(p["eps_spent_max_rdp"], 3), steps=f"{p['actual_steps']}/{p['planned_steps']}")
    if (d / "comms.json").exists():
        c = J(d / "comms.json")
        r.update(MB=round(c["MB_total"], 2), saving_pct=round(c["saving_vs_fp32_pct"], 1))
    r["mia_auc"] = round(J(d / "mia.json")["mia_auc"], 3)
    rows.append(r)
    check(f"{t}: VAL AUROC finite and >= 0.45", np.isfinite(v["mean5_auroc"]) and v["mean5_auroc"] >= 0.45, f"{v['mean5_auroc']:.4f}")
    check(f"{t}: no TEST output", not (d / "metrics_test.json").exists() and not (d / "preds_test.parquet").exists())
    f = J(d / "fairness.json")
    check(f"{t}: fairness report (subgroups, clients, F2)", {"sex", "age", "projection"} <= set(f["subgroups_val"]) and bool(f["clients_localval"])
          and "group_thresholds" in f["F2_group_thresholds"])
print()
print(pd.DataFrame(rows).to_string(index=False))
print()

if runs.get("E4_eps3"):
    p = J(runs["E4_eps3"] / "privacy.json")
    check("E4 eps: spent PRV <= target 3", p["eps_spent_max_prv"] <= 3 + 1e-6, f"{p['eps_spent_max_prv']:.4f}")
    check("E4 eps: equal across clients (PRV spread < 0.05)", max(p["eps_spent_prv"]) - min(p["eps_spent_prv"]) < 0.05, str(np.round(p["eps_spent_prv"], 3).tolist()))
    check("E4 eps: PRV <= RDP for every client", all(a <= b + 1e-9 for a, b in zip(p["eps_spent_prv"], p["eps_spent_rdp"])),
          f"PRV {p['eps_spent_max_prv']:.3f} RDP {p['eps_spent_max_rdp']:.3f}")
if runs.get("E3") and runs.get("E8_upint8_dnfp16"):
    a = pd.read_parquet(runs["E3"] / "preds_val.parquet")
    b = pd.read_parquet(runs["E8_upint8_dnfp16"] / "preds_val.parquet")
    lc = [c for c in a.columns if c.startswith("logit_")]
    diff = np.abs(a[lc].to_numpy() - b[lc].to_numpy())
    check("E8 int8 VAL predictions differ from fp32 (E3)", diff.max() > 0, f"max |dlogit| {diff.max():.2e}, mean {diff.mean():.2e}")
    c = J(runs["E8_upint8_dnfp16"] / "comms.json")
    check("E8 int8/fp16 bytes drop ~60%", 55 <= c["saving_vs_fp32_pct"] <= 70, f"{c['saving_vs_fp32_pct']:.1f}% ({c['MB_total']:.2f} MB vs {c['MB_fp32_same_rounds']:.2f} MB fp32)")
if runs.get("E8_eps3_uptopk_dnfp16"):
    c = J(runs["E8_eps3_uptopk_dnfp16"] / "comms.json")
    check("E8 top-k bytes measured", c["saving_vs_fp32_pct"] > 40, f"{c['saving_vs_fp32_pct']:.1f}%")
check("TEST never accessed in smoke", not (ROOT / "test_access.jsonl").exists())
print("\nSMOKE CHECKS", "FAILED: " + ", ".join(FAILS) if FAILS else "PASSED")
sys.exit(1 if FAILS else 0)
