"""Presentation set: demo X-rays grouped by what they show, with the exact inputs to type and the expected output.

  python scripts/make_demo_groups.py [--models models/v2.1-large] [--out demo/presentation]

Reads demo/demo_samples.csv (from scripts/make_demo_set.py), runs every image through the exported private (E4, eps = 3), FedAvg (E3)
and central (E2) models with the dashboard's own pipeline (dpmmfl.serve.Predictor), and copies selected images into groups:

  1_normal_vs_abnormal      a normal chest next to clearly abnormal ones
  2_one_finding_each        one image per finding the private model detects reliably (VAL AUROC >= 0.70)
  3_multiple_findings       one image, several findings detected at once
  4_private_vs_nonprivate   central and private model agree / only the central model detects (cost of privacy)
  5_membership_privacy      training-population vs held-out patients with the same finding (Privacy tab)
  6_honest_misses           findings the private model misses on randomly chosen patients

Groups 1-4 and 6 use held-out VAL patients only; TEST is never read. "Detected" means probability >= the model's VAL Youden threshold
(calibration.json). Selected cases are chosen because they work and overstate accuracy; group 6 and the random cases show typical behaviour.
Writes <out>/README.md (demo script), <out>/predictions.csv and the copied images.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dpmmfl.constants import LABELS  # noqa: E402
from dpmmfl.serve import Predictor  # noqa: E402

MODELS = {"private": "E4_eps3_s0", "fedavg": "E3_s0", "central": "E2_s0"}
NAMES = {"private": "Private (DP, eps = 3)", "fedavg": "FedAvg (no DP)", "central": "Central (no DP)"}
SHORT = {"No Finding": "NoFinding", "Enlarged Cardiomediastinum": "EnlCardiomed", "Lung Opacity": "LungOpacity", "Lung Lesion": "LungLesion",
         "Pleural Effusion": "PlEffusion", "Pleural Other": "PlOther", "Support Devices": "SupportDev"}


def short(lab):
    return SHORT.get(lab, lab.replace(" ", ""))


def predict_all(samples, model_root):
    """One row per (image, model, label): prob, threshold, detected."""
    rows = []
    for key, name in MODELS.items():
        md = model_root / name
        P = Predictor(md)
        thr = json.loads((md / "calibration.json").read_text())["thresholds"]
        for r in samples.itertuples():
            gray = np.asarray(Image.open(ROOT / r.image).convert("L"))
            df, _ = P.predict(gray, age=float(r.age), sex=r.sex, projection=r.projection)
            for lab, p in zip(df.label, df.prob):
                rows.append(dict(sample_id=r.sample_id, model=key, label=lab, prob=float(p), thr=float(thr[lab]), detected=bool(p >= thr[lab]),
                                 val_auroc=P.val_auroc.get(lab)))
        print(f"{name}: predicted {len(samples)} images")
    return pd.DataFrame(rows)


def truth(row):
    """Report labels of one sample: (positives, uncertain)."""
    pos = [l for l in LABELS if row.get(f"label_{l}") == 1]
    unc = [l for l in LABELS if row.get(f"label_{l}") == -1]
    return pos, unc


def summarise(samples, P):
    out = []
    for _, r in samples.iterrows():
        pos, unc = truth(r)
        rec = dict(sample_id=r.sample_id, finding=r.finding, kind=r.kind, split=r.split, image=r.image, patient_id=r.patient_id,
                   age=int(r.age), sex=r.sex, projection=r.projection, positives=pos, uncertain=unc)
        for m in MODELS:
            q = P[(P.sample_id == r.sample_id) & (P.model == m)].set_index("label")
            det = [l for l in LABELS if q.loc[l, "detected"]]
            rec[f"{m}_detected"] = det
            rec[f"{m}_tp"] = [l for l in pos if l in det]
            rec[f"{m}_fn"] = [l for l in pos if l not in det]
            rec[f"{m}_fp"] = [l for l in det if l not in pos and l not in unc]
            rec[f"{m}_prob"] = q.prob.to_dict()
            rec[f"{m}_thr"] = q.thr.to_dict()
            rec[f"{m}_margin"] = (q.prob / q.thr).to_dict()          # > 1 means detected; larger = more confident relative to the threshold
            rec[f"{m}_alarms"] = [l for l in LABELS if q.loc[l, "prob"] >= 0.5 and l not in pos and l not in unc]   # confident false alarms
        out.append(rec)
    return pd.DataFrame(out)


def pick(S, used, cond, key, n, ascending=False):
    """Up to n rows of S satisfying cond, sorted by key, not used before."""
    c = S[S.apply(cond, axis=1) & ~S.sample_id.isin(used)]
    if c.empty:
        return c
    c = c.assign(_k=c.apply(key, axis=1)).sort_values("_k", ascending=ascending).head(n)
    used.update(c.sample_id)
    return c.drop(columns="_k")


def alarms(r):
    """Confident false alarms of the private and central models (penalised when choosing showcase images)."""
    return len(r.private_alarms) + len(r.central_alarms)


def strong_tp(r, p=0.4):
    """Report findings the private model detects with a clearly visible probability (>= p)."""
    return [l for l in r.private_tp if r.private_prob[l] >= p]


def build_groups(S, P):
    held = S[S.split == "val"]
    used = set()
    G = {}
    auroc = {l: P[(P.model == "private") & (P.label == l)].val_auroc.iloc[0] or 0 for l in LABELS}
    reliable = [l for l in LABELS if l != "No Finding" and auroc[l] >= 0.70]
    well_known = [l for l in LABELS if l != "No Finding" and auroc[l] >= 0.65]          # for the private vs non-private comparison

    # 1 normal vs abnormal
    normal = pick(held, used, lambda r: r.positives == ["No Finding"] and "No Finding" in r.private_detected and alarms(r) == 0,
                  lambda r: r.private_prob["No Finding"], 2)
    G["1_normal_vs_abnormal"] = [("normal", r) for _, r in normal.iterrows()]      # abnormal examples are added after groups 2 and 3

    # 2 one image per reliably detected finding
    G["2_one_finding_each"] = []
    for lab in reliable:
        key = lambda r, lab=lab: r.private_prob[lab] - 0.3 * alarms(r)
        c = pick(held, used, lambda r, lab=lab: lab in r.private_tp and r.finding == lab and not r.private_alarms, key, 1)
        if c.empty:
            c = pick(held, used, lambda r, lab=lab: lab in r.private_tp and not r.private_alarms, key, 1)
        G["2_one_finding_each"] += [(short(lab), r) for _, r in c.iterrows()]

    # 3 several findings in one image
    multi = pick(held, used, lambda r: len(strong_tp(r, 0.3)) >= 2 and not r.private_alarms,
                 lambda r: len(strong_tp(r, 0.3)) + 0.5 * sum(r.private_prob[l] for l in strong_tp(r, 0.3)) - 0.5 * alarms(r), 3)
    G["3_multiple_findings"] = [(f"{len(strong_tp(r, 0.3))}findings", r) for _, r in multi.iterrows()]

    # 1 (continued) abnormal examples from the remaining images
    abnormal = pick(held, used, lambda r: len(strong_tp(r, 0.3)) >= 2 and "No Finding" not in r.positives and not r.private_alarms,
                    lambda r: sum(r.private_prob[l] for l in strong_tp(r, 0.3)) - alarms(r), 2)
    if abnormal.empty or len(abnormal) < 2:
        abnormal = pd.concat([abnormal, pick(held, used, lambda r: len(strong_tp(r)) >= 1 and "No Finding" not in r.positives and not r.private_alarms,
                                             lambda r: sum(r.private_prob[l] for l in strong_tp(r)) - alarms(r), 2 - len(abnormal))])
    G["1_normal_vs_abnormal"] += [("abnormal", r) for _, r in abnormal.iterrows()]

    # 4 private vs non-private on the image's own finding (findings both models handle reasonably well)
    agree = pick(held, used, lambda r: r.finding in well_known and r.finding in r.private_tp and r.finding in r.central_tp,
                 lambda r: r.private_prob[r.finding] - 0.3 * alarms(r), 2)
    cost = pick(held, used, lambda r: r.finding in well_known and r.finding in r.central_tp and r.finding in r.private_fn,
                lambda r: r.central_prob[r.finding] - 0.3 * alarms(r), 2)
    G["4_private_vs_nonprivate"] = [("both_detect", r) for _, r in agree.iterrows()] + [("only_central", r) for _, r in cost.iterrows()]

    # 5 membership pairs: training-population patient vs held-out patient with the same finding
    G["5_membership_privacy"] = []
    for _, t in S[S.kind == "training"].iterrows():
        h = held[(held.finding == t.finding) & (held.kind == "random") & ~held.sample_id.isin(used)].head(1)
        if len(h) and len(G["5_membership_privacy"]) < 6:
            G["5_membership_privacy"] += [(f"member_{short(t.finding)}", t), (f"heldout_{short(t.finding)}", h.iloc[0])]
            used.update([t.sample_id, h.iloc[0].sample_id])

    # 6 honest misses on random patients
    miss = pick(held, used, lambda r: r.kind == "random" and r.finding in r.private_fn and r.finding != "No Finding",
                lambda r: r.private_margin[r.finding], 3, ascending=True)
    G["6_honest_misses"] = [(f"missed_{short(r.finding)}", r) for _, r in miss.iterrows()]
    return G, reliable


def fmt_truth(r, m):
    """The model's probability for each finding in the report: check = detected, cross = missed."""
    labs = [l for l in r.positives] or ["No Finding"]
    return ", ".join(f"{l} {r[f'{m}_prob'][l]:.2f} {'✓' if l in r[f'{m}_detected'] else '✗'}" for l in labs)


def fmt_alarm(r, m):
    """Strongest confident false alarm (probability >= 0.5 on a finding not in the report), if any."""
    a = sorted(r[f"{m}_alarms"], key=lambda l: -r[f"{m}_prob"][l])
    return f"{a[0]} {r[f'{m}_prob'][a[0]]:.2f}" + (f" (+{len(a) - 1})" if len(a) > 1 else "") if a else "none"


GROUP_TEXT = {
    "1_normal_vs_abnormal": ("Normal vs abnormal", "Start here. Show a normal chest (high *No Finding*), then abnormal chests where the private model flags real findings."),
    "2_one_finding_each": ("One finding at a time", "One image per finding that the private model detects reliably (VAL AUROC at least 0.70). "
                           "Pick the finding, open the Grad-CAM for it and point at the highlighted region."),
    "3_multiple_findings": ("Several findings in one X-ray", "Multi-label classification: the model scores all 14 findings at once, and several are detected together."),
    "4_private_vs_nonprivate": ("Private vs non-private model", "Switch the model between *Central* and *E4 eps=3* on the same image. "
                                "`both_detect`: privacy costs nothing here. `only_central`: the cost of privacy, which is largest for subtle and rare findings."),
    "5_membership_privacy": ("Membership privacy", "Pairs of a training-population patient and a held-out patient with the same finding. Use them to explain what "
                             "membership inference asks (was this patient in training?), then run the live attack on the Privacy tab: the central model "
                             "leaks measurably, the private model stays at chance. A single image pair does not show leakage by itself."),
    "6_honest_misses": ("Honest misses", "Randomly chosen patients whose finding the private model misses. Show these to make clear the model is a research "
                        "prototype and not a diagnostic tool."),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="models/v2.1-large")
    ap.add_argument("--out", default="demo/presentation")
    a = ap.parse_args()
    samples = pd.read_csv(ROOT / "demo/demo_samples.csv")
    model_root, out = ROOT / a.models, ROOT / a.out
    for m in MODELS.values():
        assert (model_root / m / "head.pt").exists() and (model_root / m / "DONE").exists(), f"model not ready: {model_root / m}"

    P = predict_all(samples, model_root)
    S = summarise(samples, P)
    G, reliable = build_groups(S, P)

    if out.exists():
        shutil.rmtree(out)
    lines = ["# Demo presentation set", "",
             "X-rays grouped by what they demonstrate. Each table gives the inputs to type on the dashboard's **Predict** tab and what the "
             "models output. Start the dashboard with `streamlit run dashboard/app.py`, choose **Trained on: saved (v2.1-large)** and the model, "
             "upload the image and enter age, sex and view exactly as listed.", "",
             "- **Private** = `E4_eps3` (patient-level DP, eps = 3). **Central** = `E2` (no privacy). **FedAvg** = `E3` (federated, no DP).",
             "- The model columns give the calibrated probability for each finding in the radiology report: ✓ = at or above the model's decision "
             "threshold (fitted on validation patients), ✗ = below it. *False alarm* = the strongest finding not in the report with probability of at least 0.5.",
             "- Rare findings (fracture, lung lesion, pleural other, enlarged cardiomediastinum) have very low thresholds, so the dashboard may flag them "
             "at small probabilities. Focus the demo on the report findings and their probabilities.",
             "- Groups 1-5 contain **selected** cases that work; they overstate accuracy. Group 6 shows typical misses. All patients except the "
             "`member_*` images in group 5 are held-out validation patients the models never trained on. No TEST image is used.",
             "- Research prototype, not for clinical use.", ""]
    rows = []
    for g, items in G.items():
        title, text = GROUP_TEXT[g]
        (out / g).mkdir(parents=True, exist_ok=True)
        lines += [f"## {g.split('_', 1)[0]}. {title}", "", text, "",
                  "| image | age | sex | view | report findings | private model | private false alarm | central model | central false alarm |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for i, (tag, r) in enumerate(items, 1):
            fname = f"{i:02d}_{tag}_{r.patient_id}.jpg"
            shutil.copy2(ROOT / r.image, out / g / fname)
            lab = ", ".join(r.positives) or "none"
            if r.uncertain:
                lab += f" (uncertain: {', '.join(r.uncertain)})"
            lines.append(f"| `{g}/{fname}` | {r.age} | {r.sex} | {r.projection} | {lab} | {fmt_truth(r, 'private')} | {fmt_alarm(r, 'private')} "
                         f"| {fmt_truth(r, 'central')} | {fmt_alarm(r, 'central')} |")
            rows.append(dict(group=g, file=f"{g}/{fname}", patient_id=r.patient_id, kind=r.kind, age=r.age, sex=r.sex, projection=r.projection,
                             report_positives="; ".join(r.positives), report_uncertain="; ".join(r.uncertain),
                             **{f"{m}_detected": "; ".join(r[f"{m}_detected"]) for m in MODELS},
                             **{f"{m}_true_positives": "; ".join(r[f"{m}_tp"]) for m in MODELS},
                             **{f"{m}_missed": "; ".join(r[f"{m}_fn"]) for m in MODELS}))
        lines.append("")
    lines += ["## Model readiness check", "",
              f"All {len(samples)} demo images were scored by the three exported models with the dashboard's pipeline. "
              f"Findings with private-model VAL AUROC of at least 0.70 (used for group 2): {', '.join(reliable)}.", ""]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    print(f"wrote {out} ({len(rows)} images in {len(G)} groups)")


if __name__ == "__main__":
    main()
