"""Organise the demo images by finding, with patient details, a short description and the saved models' predictions.

  python scripts/make_demo_folders.py [--models models/v2.1-large] [--out demo/by_label]

Reads demo/demo_samples.csv (from scripts/make_demo_set.py). For every finding writes demo/by_label/<NN>_<finding>/ with the images,
info.md (what the finding is, each patient's details, CheXpert labels, model output) and a top-level README.md + predictions.csv.
Predictions use the same pipeline as the dashboard (dpmmfl.serve.Predictor). "Detected" means probability >= the run's VAL Youden
threshold (calibration.json). TEST is never read.
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
from dpmmfl.constants import COMP5, LABELS  # noqa: E402

ABOUT = {
    "No Finding": "No abnormality labelled on the study. Used as the 'normal' class; the model should give it a high probability and "
                  "low probabilities to the other findings.",
    "Enlarged Cardiomediastinum": "Widening of the central chest shadow (heart and mediastinum). Can point to aortic disease, a mass or "
                                  "fluid. A subtle, subjective label, and one of the hardest for every model.",
    "Cardiomegaly": "Enlarged heart: the heart is wider than about half the chest on a PA view. Seen in heart failure and cardiomyopathy. "
                    "AP (bedside) views make the heart look larger.",
    "Lung Opacity": "Any area of the lung that looks whiter (denser) than it should. An umbrella term covering edema, consolidation, "
                    "atelectasis, masses.",
    "Lung Lesion": "A focal nodule or mass in the lung, benign or malignant. Rare in the data, so harder to learn.",
    "Edema": "Fluid in the lung tissue, usually from heart failure. Hazy opacities spreading from both hila, sometimes thin horizontal "
             "lines at the lung edges.",
    "Consolidation": "Air spaces filled with fluid, pus or cells, giving a dense white area, often with visible air-filled bronchi. "
                     "Typical of pneumonia.",
    "Pneumonia": "Lung infection. Radiologically it looks like consolidation; the label partly depends on clinical information, so it "
                 "is noisy.",
    "Atelectasis": "Partial collapse of lung tissue: band-like or plate-like opacities with volume loss, usually at the lung bases.",
    "Pneumothorax": "Air between the lung and chest wall: a thin pleural line with no lung markings beyond it. Can be an emergency.",
    "Pleural Effusion": "Fluid around the lung: the sharp angle at the base of the lung becomes blunted, with a curved fluid line.",
    "Pleural Other": "Other pleural abnormalities (thickening, plaques, scarring). Rare and heterogeneous.",
    "Fracture": "A broken bone, usually a rib or clavicle. Often subtle on a chest X-ray.",
    "Support Devices": "Medical hardware: tubes, central lines, pacemakers, ECG leads. Usually easy to see.",
}
SOURCE = {"showcase": "held-out VAL patient (never seen by the models), selected because the DP model detects the finding",
          "random": "held-out VAL patient, picked at random (never seen by the models)",
          "training": "training-pool patient: used in training (membership-inference demo)"}
MODELS = (("E4_eps3_s0","DP model (ε = 3)"), ("E2_s0", "central model (no privacy)"))


def label_state(row, lab):
    v = row.get(f"label_{lab}")
    return "positive" if v == 1 else "uncertain" if v == -1 else "negative" if v == 0 else "not mentioned"


def add_showcase(s, model_dir, k):
    """Add k 'showcase' images per finding to demo_samples.csv: held-out VAL patients whose finding the model detects (stored VAL
    predictions of the exported run, prob >= Youden threshold, finding in the model's top 3), few co-occurring findings. Random among
    those, seed 0. Clearly labelled as selected; the random held-out images stay alongside."""
    if (s.kind == "showcase").any():
        return s
    run_id = json.loads((model_dir / "manifest.json").read_text())["run_id"]
    pv = pd.read_parquet(ROOT / "results/v2.1-final/runs" / run_id / "preds_val.parquet")
    thr = json.loads((model_dir / "calibration.json").read_text())["thresholds"]
    cols = ["row_id", "patient_id", "study_num", "Sex", "Age", "AP/PA", "abs_path"] + [f"R_{l}" for l in LABELS]
    m = pd.read_parquet(ROOT / "cache/meta_train.parquet", columns=cols).merge(pv, on=["row_id", "patient_id"])
    probs = m[[f"prob_{l}" for l in LABELS]].to_numpy()
    m["n_pos"] = (m[[f"R_{l}" for l in LABELS]] == 1).sum(1)
    m = m[m.Sex.isin(["Male", "Female"]) & m.Age.notna() & m["AP/PA"].isin(["AP", "PA"])]
    used, rows, rng = set(s.patient_id), [], np.random.default_rng(0)
    for j, lab in enumerate(LABELS):
        rank = (probs[m.index] > probs[m.index, j:j + 1]).sum(1) + 1
        c = m[(m[f"R_{lab}"] == 1) & (m[f"prob_{lab}"] >= thr[lab]) & (rank <= 3) & (m.n_pos <= (1 if lab == "No Finding" else 3))]
        c = c[~c.patient_id.isin(used)].drop_duplicates("patient_id")
        for _, r in c.iloc[rng.permutation(len(c))[:k]].iterrows():
            used.add(r.patient_id)
            dest = ROOT / "demo/images" / f"{lab.replace(' ', '_')}_{r.patient_id}_showcase.jpg"
            shutil.copy(r.abs_path, dest)
            rec = dict(sample_id=dest.stem, finding=lab, image=f"demo/images/{dest.name}", patient_id=r.patient_id, study=r.study_num,
                       sex=r.Sex, age=int(r.Age), projection=r["AP/PA"], source="held-out (VAL, never trained on), showcase",
                       split="val", kind="showcase")
            rec.update({f"label_{l}": r[f"R_{l}"] for l in LABELS})
            rows.append(rec)
    s = pd.concat([pd.DataFrame(rows), s], ignore_index=True)
    s.to_csv(ROOT / "demo/demo_samples.csv", index=False)
    print(f"added {len(rows)} showcase images to demo/demo_samples.csv")
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", default="models/v2.1-large")
    ap.add_argument("--out", default="demo/by_label")
    ap.add_argument("--showcase", type=int, default=3, help="selected held-out images per finding")
    a = ap.parse_args()
    from dpmmfl.serve import Predictor
    import torch
    s = pd.read_csv(ROOT / "demo/demo_samples.csv")
    if "kind" not in s:
        s["kind"] = np.where(s.split == "val", "random", "training")
    s = add_showcase(s, ROOT / a.models / MODELS[0][0], a.showcase)
    out = ROOT / a.out
    if out.exists():
        shutil.rmtree(out)                                                       # generated by this script only
    out.mkdir(parents=True)

    preds, info = {}, {}
    for key, _ in MODELS:
        md = ROOT / a.models / key
        P = Predictor(md)
        thr = json.loads((md / "calibration.json").read_text())["thresholds"]
        info[key] = dict(thr=thr, auroc=P.val_auroc)
        for r in s.itertuples():
            gray = np.array(Image.open(ROOT / r.image).convert("L"))
            df, _ = P.predict(gray, age=float(r.age), sex=r.sex, projection=r.projection)
            preds[(key, r.sample_id)] = df.set_index("label").prob
        del P
        torch.cuda.empty_cache()
        print("predicted with", key)

    rows, index = [], []
    for n, lab in enumerate(LABELS, 1):
        d = out / f"{n:02d}_{lab.replace(' ', '_')}"
        d.mkdir(exist_ok=True)
        order = {"showcase": 0, "random": 1, "training": 2}
        sub = s[s.finding == lab].sort_values("kind", key=lambda k: k.map(order), kind="stable")
        md = [f"# {lab}" + ("  (one of the 5 CheXpert competition findings)" if lab in COMP5 else ""), "", ABOUT[lab], "",
              "| model | VAL AUROC for this finding | decision threshold |", "|---|---|---|"]
        for key, name in MODELS:
            au = info[key]["auroc"].get(lab)
            md.append(f"| {name} | {au:.3f} | {info[key]['thr'][lab]:.2f} |" if au == au and au is not None else f"| {name} | – | – |")
        md += ["", "AUROC 0.5 = chance, 1.0 = perfect. A finding counts as detected when its probability is at or above the threshold.", ""]
        best = None
        for r in sub.itertuples():
            held = r.split == "val"
            i = (sub.kind[:sub.index.get_loc(r.Index) + 1] == r.kind).sum()
            fname = f"{dict(showcase='showcase', random='heldout_random', training='training')[r.kind]}_{i}_{r.patient_id}.jpg"
            shutil.copy2(ROOT / r.image, d / fname)
            row = s.loc[s.sample_id == r.sample_id].iloc[0]
            truth = [l for l in LABELS if label_state(row, l) == "positive"]
            unc = [l for l in LABELS if label_state(row, l) == "uncertain"]
            md += [f"## {fname}", "", f"- **Patient:** {r.patient_id}, {r.sex}, age {int(r.age)}, {r.projection} view",
                   f"- **Source:** {SOURCE[r.kind]}",
                   f"- **CheXpert labels (radiology report):** positive: {', '.join(truth) or 'none'}" + (f"; uncertain: {', '.join(unc)}" if unc else "")]
            rec = dict(finding=lab, file=f"{d.name}/{fname}", patient_id=r.patient_id, sex=r.sex, age=int(r.age), projection=r.projection,
                       source=r.kind, true_positive=";".join(truth))
            for key, name in MODELS:
                p = preds[(key, r.sample_id)]
                t = info[key]["thr"][lab]
                rank = int((p > p[lab]).sum()) + 1
                top3 = ", ".join(f"{l} {v:.2f}" for l, v in p.sort_values(ascending=False).head(3).items())
                det = p[lab] >= t
                md.append(f"- **{name}:** P({lab}) = {p[lab]:.2f} (threshold {t:.2f}) → **{'detected' if det else 'not detected'}**; "
                          f"rank {rank} of 14; top 3: {top3}")
                rec.update({f"{key}_prob": round(float(p[lab]), 3), f"{key}_threshold": round(t, 3), f"{key}_detected": bool(det), f"{key}_rank": rank})
                if key == MODELS[0][0] and held and det and (best is None or p[lab] / t > best[1]):
                    best = (fname, p[lab] / t, float(p[lab]))
            md.append("")
            rows.append(rec)
        md.insert(1, "")
        md.insert(2, f"**Best image to demo:** `{best[0]}` (DP model P = {best[2]:.2f})" if best else
                  "**Note:** the DP model does not detect this finding on any held-out example here; use it to show a limitation.")
        (d / "info.md").write_text("\n".join(md) + "\n", encoding="utf8")
        hits = sum(r[f"{MODELS[0][0]}_detected"] for r in rows if r["finding"] == lab and r["source"] == "random")
        nh = sum(1 for r in rows if r["finding"] == lab and r["source"] == "random")
        index.append(f"| [{lab}]({d.name}/info.md) | {'yes' if lab in COMP5 else ''} | {info[MODELS[0][0]]['auroc'].get(lab, float('nan')):.3f} | "
                     f"{hits} / {nh} | {('`' + best[0] + '`') if best else '–'} |")

    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
    readme = ["# Demo images by finding", "",
              "One folder per CheXpert finding. Every folder has three kinds of X-ray:", "",
              "- `showcase_*`: held-out patients (never seen by the models) **selected** because the DP model detects the finding.",
              "  Use these to show the model working. They are chosen, so they overstate accuracy.",
              "- `heldout_random_*`: held-out patients picked at random. These show typical behaviour, misses included.",
              "- `training_*`: a patient from the training population (used in the membership-inference demo).", "",
              "`info.md` in each folder explains the finding and lists every patient's age, sex and view, the radiology-report labels and",
              f"what the saved large models (`{a.models}`) predict. All predictions are in `predictions.csv`.", "",
              "| finding | competition 5 | DP model VAL AUROC | random held-out detected (DP model) | best image to demo |", "|---|---|---|---|---|",
              *index, "",
              "## How to demo", "",
              "1. `streamlit run dashboard/app.py`, open **Predict**, choose the model **saved (v2.1-large) / E4_eps3** (the private model).",
              "2. Open a finding's `info.md`, upload the best image, and type the patient's age, sex and view from that file",
              "   (or click **Use** on the same image in the example gallery, which fills them in).",
              "3. The finding should be among the highest bars; Grad-CAM shows where the model looked.",
              "4. Show a **No Finding** image next: the disease bars drop.",
              "5. Compare with the central model (E2) to show what privacy costs: usually a few points of probability.", "",
              "Detection is only as good as the per-finding AUROC above: strong for Pleural Effusion, Edema, Cardiomegaly, Support Devices",
              "and No Finding; weak for rare or subtle findings (Lung Lesion, Fracture, Enlarged Cardiomediastinum). Labels come from",
              "automatic report parsing, so some are noisy. Demonstration only, not for clinical use.", ""]
    (out / "README.md").write_text("\n".join(readme), encoding="utf8")
    print(f"wrote {len(rows)} images in {len(LABELS)} folders -> {out}")


if __name__ == "__main__":
    main()
