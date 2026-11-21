"""DP-MMFL dashboard: results, model predictions on chest X-rays, and a membership-inference check.

    streamlit run dashboard/app.py

Reads results/ (v2.1 scales, results/v2-large/ for the v2 full-data run, results/mia_*/ for the attacks) and models/. The only file it writes is
demo/patient_records.jsonl, when a record is saved on the Predict tab.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dpmmfl.constants import COMP5, LABELS                      # noqa: E402
from dpmmfl.runs import discover_runs                          # noqa: E402  (torch is imported lazily)

st.set_page_config(page_title="DP-MMFL", layout="wide")

NAMES = {"E0": "Central, linear probe", "E1": "Central, image only", "E2": "Central, with metadata", "E3": "FedAvg",
         "E4_eps8": "DP ε=8", "E4_eps3": "DP ε=3", "E4_eps1": "DP ε=1", "E5_eps8": "Image-level DP ε=8",
         "E7_eps3_F1": "DP ε=3, reweighted", "E8_upint8_dnfp16": "FedAvg, int8 up / fp16 down"}
V2_TAGS = {"E3_upint8_dnfp16": "E8_upint8_dnfp16"}                 # archived v2 tag -> current tag
ORDER = list(NAMES.values())
EPS = {"FedAvg": np.inf, "DP ε=8": 8, "DP ε=3": 3, "DP ε=1": 1}
SCALES = ("large (v2.1)", "medium", "small")
DEMO = ROOT / "demo"
RECORDS = DEMO / "patient_records.jsonl"
COLORS = {"good": "#2a9d8f", "moderate": "#e9c46a", "weak": "#b8b8b8"}


def fmt(v, spec=".3f"):
    return "-" if v is None or pd.isna(v) else format(v, spec)


# ------------------------------------------------------------------ data
@st.cache_data(show_spinner=False)
def load_scales():
    """scale name -> per-config table (mean over seeds of the final runs)."""
    tabs = {}
    for name, f in (("small", "results/scale_small"), ("medium", "results/scale_medium"), ("large (v2.1)", "results/v2.1-final")):
        p = ROOT / f / "master_runs.csv"
        if not p.exists():
            continue
        d = pd.read_csv(p)
        d = d[d.final.astype(str) == "True"]
        if not len(d):
            continue
        g = d.groupby("tag")
        t = g.mean(numeric_only=True)
        t["n_seeds"] = g.size()
        t["val_sd"], t["test_sd"] = g.val_mean5.std(), g.test_mean5.std()
        tabs[name] = t.rename(index=NAMES)
    v2 = ROOT / "results/v2-large/summary.csv"
    if v2.exists():
        s = pd.read_csv(v2).set_index("run").rename(index=V2_TAGS)
        t = pd.DataFrame({"val_mean5": s.val, "test_mean5": s.test, "test_sd": s.test_std, "mia_auc": s.mia_auc, "n_seeds": s.n})
        tabs["large (v2)"] = t.rename(index=NAMES)
    return tabs


@st.cache_data(show_spinner=False)
def runs_table():
    d = discover_runs()
    return d[d.predictable & d.tag.isin(NAMES)] if len(d) else d


@st.cache_resource(show_spinner="Loading model…")
def get_predictor(path):
    from dpmmfl.serve import Predictor
    return Predictor(path)


@st.cache_data(show_spinner=False)
def load_demo():
    f = DEMO / "demo_samples.csv"
    if not f.exists():
        return pd.DataFrame()
    d = pd.read_csv(f)
    return d[d.image.map(lambda p: (ROOT / p).exists())].reset_index(drop=True)   # images are not redistributed: use only those present


def scale_options(rt):
    """Known scales first (largest first), then saved models."""
    return [s for s in SCALES if s in set(rt.scale)] + sorted(set(rt.scale) - set(SCALES))


def model_picker(rt, key, default_tag="E4_eps3"):
    """Scale / setup / seed selectors. Returns the chosen run row."""
    a, b, c = st.columns(3)
    sc = a.selectbox("Trained on", scale_options(rt), key=f"{key}_scale")
    r1 = rt[rt.scale == sc]
    tags = [t for t in NAMES if t in set(r1.tag)]
    tg = b.selectbox("Setup", tags, index=tags.index(default_tag) if default_tag in tags else 0,
                     format_func=NAMES.get, key=f"{key}_tag")
    r2 = r1[r1.tag == tg].sort_values("seed")
    sd = c.selectbox("Seed", list(r2.seed), key=f"{key}_seed")
    return r2[r2.seed == sd].iloc[0]


tabs = load_scales()
rt = runs_table()
st.title("DP-MMFL")
st.caption("Chest X-ray classification with federated learning, patient-level differential privacy, compressed communication "
           "and a fairness audit. CheXpert, 14 findings.")
tab_res, tab_pred, tab_priv, tab_about = st.tabs(["Results", "Predict", "Privacy", "About"])

# ------------------------------------------------------------------ results
with tab_res:
    if not tabs:
        st.info("No results yet. Run the experiments, then `python scripts/aggregate.py`.")
    else:
        scales = list(tabs)
        c1, c2 = st.columns(2)
        metric = c1.radio("Metric", ["TEST", "VAL"], horizontal=True,
                          help="Mean AUROC over the 5 competition findings. TEST has 202 images (±0.03); VAL has thousands of patients.")
        focus = c2.selectbox("Scale", scales, index=scales.index("large (v2.1)") if "large (v2.1)" in scales else 0)
        mcol, scol = ("test_mean5", "test_sd") if metric == "TEST" else ("val_mean5", "val_sd")
        t = tabs[focus]

        def cell(cfg, col):
            return t.loc[cfg, col] if cfg in t.index and col in t.columns else None

        k = st.columns(4)
        k[0].metric("Central", fmt(cell("Central, image only", mcol)))
        k[1].metric("Federated", fmt(cell("FedAvg", mcol)))
        k[2].metric("Federated + DP (ε=3)", fmt(cell("DP ε=3", mcol)))
        sv = cell("FedAvg, int8 up / fp16 down", "saving_pct")
        k[3].metric("Traffic saved", "-" if sv is None or pd.isna(sv) else f"{sv:.0f}%")

        st.subheader("By data scale")
        L = pd.DataFrame([dict(scale=s, config=c, auroc=d.loc[c, mcol], sd=d.loc[c, scol] if scol in d else np.nan)
                          for s, d in tabs.items() for c in ORDER if c in d.index and pd.notna(d.loc[c, mcol])])
        fig = px.bar(L, x="config", y="auroc", color="scale", barmode="group", error_y="sd", category_orders={"config": ORDER},
                     labels={"auroc": f"{metric} AUROC", "config": ""})
        fig.update_yaxes(range=[0.55, 0.86] if metric == "TEST" else [0.55, 0.78])
        st.plotly_chart(fig, use_container_width=True)
        st.caption("small = 2k patients, medium = 20k, large = full data (64k). "
                   "large (v2) is the earlier code version with 30 rounds; the others use the current code.")

        left, right = st.columns(2)
        with left:
            st.subheader("Privacy vs accuracy")
            P = L[L.config.isin(EPS)].assign(x=lambda d: d.config.map(EPS).replace(np.inf, 16))
            pf = px.line(P.sort_values("x"), x="x", y="auroc", color="scale", markers=True, labels={"auroc": f"{metric} AUROC"})
            pf.update_xaxes(tickvals=[1, 3, 8, 16], ticktext=["ε=1", "ε=3", "ε=8", "no DP"], title="")
            st.plotly_chart(pf, use_container_width=True)
            st.caption("Smaller ε = stronger privacy. The cost shrinks as data grows.")
        with right:
            st.subheader("Communication")
            if "MB_total" in t.columns and t.MB_total.notna().any():
                cm = t.loc[[i for i in ORDER if i in t.index], ["MB_total", "saving_pct"]].dropna(subset=["MB_total"])
                cf = px.bar(cm.reset_index(names="config"), x="config", y="MB_total", text="saving_pct",
                            labels={"MB_total": "MB sent per training run", "config": ""})
                cf.update_traces(texttemplate="%{text:.0f}% saved")
                st.plotly_chart(cf, use_container_width=True)
                st.caption("Bytes measured on the wire, all clients, both directions.")
            else:
                st.info("No communication data for this scale.")

        st.subheader("Fairness (VAL)")
        fl, fr = st.columns(2)
        fcols = {"gap_sex": "sex", "gap_age": "age", "gap_projection": "projection"}
        if all(c in t.columns for c in fcols) and t.gap_sex.notna().any():
            F = t[list(fcols)].dropna().rename(columns=fcols).loc[lambda d: d.index.isin(ORDER)]
            F = F.reset_index(names="config").melt("config", var_name="group", value_name="gap")
            fl.plotly_chart(px.bar(F, x="config", y="gap", color="group", barmode="group", category_orders={"config": ORDER},
                                   labels={"gap": "AUROC gap between groups", "config": ""}), use_container_width=True)
            if "F2_tpr_gap_global" in t.columns and t.F2_tpr_gap_global.notna().any():
                g = t[["F2_tpr_gap_global", "F2_tpr_gap_group"]].dropna().loc[lambda d: d.index.isin(ORDER)]
                gf = go.Figure([go.Bar(name="one threshold", x=g.index, y=g.F2_tpr_gap_global),
                                go.Bar(name="per-group thresholds", x=g.index, y=g.F2_tpr_gap_group)])
                gf.update_layout(barmode="group", yaxis_title="sensitivity gap, AP vs PA")
                fr.plotly_chart(gf, use_container_width=True)
            st.caption("Left: largest AUROC difference between groups. Right: per-group decision thresholds "
                       "(fit on half of VAL, measured on the other half) close most of the sensitivity gap.")
        else:
            st.info("No fairness data for this scale.")

        with st.expander("Per-finding AUROC of one run"):
            if len(rt):
                row = model_picker(rt, "pl", "E3")
                mv, mt = (json.loads((Path(row.path) / f).read_text()) for f in ("metrics_val.json", "metrics_test.json"))
                pl = pd.DataFrame({"finding": LABELS, "VAL": [mv["auroc"].get(x) for x in LABELS], "TEST": [mt["auroc"].get(x) for x in LABELS]})
                pf = px.bar(pl.melt("finding", var_name="split", value_name="AUROC"), x="finding", y="AUROC", color="split", barmode="group")
                pf.add_hline(y=0.5, line_dash="dot")
                st.plotly_chart(pf, use_container_width=True)
                st.caption(f"Competition findings: {', '.join(COMP5)}. Rare findings have few TEST positives, so their TEST values are noisy.")
        with st.expander("Tables"):
            for s, d in tabs.items():
                st.markdown(f"**{s}**")
                st.dataframe(d.round(4), use_container_width=True)

# ------------------------------------------------------------------ predict
with tab_pred:
    if not len(rt):
        st.info("No trained models found.")
    else:
        row = model_picker(rt, "pred")
        st.caption(f"Model accuracy: VAL {fmt(row.val)} · TEST {fmt(row.test)} (mean AUROC, 5 competition findings)")
        demo = load_demo()
        ss = st.session_state
        inp, out = st.columns([1, 1.4], gap="large")

        with inp:
            source = st.radio("Image", ["Example", "Upload"], horizontal=True) if len(demo) else "Upload"
            image, rec = None, None
            if source == "Example":
                findings = sorted(demo.finding.unique())
                finding = st.selectbox("Finding", findings, index=findings.index("Cardiomegaly") if "Cardiomegaly" in findings else 0)
                pool = demo[demo.finding == finding].reset_index(drop=True)
                use = lambda r: ss.update(sample=r.sample_id, p_id=r.patient_id, p_age=int(r.age), p_sex=r.sex, p_proj=r.projection, p_note="")
                if ss.get("sample") not in set(pool.sample_id):  # new finding: start with its first image
                    use(pool.iloc[0])
                for i, r in pool.iterrows():
                    if i % 4 == 0:
                        cols = st.columns(4)
                    cols[i % 4].image(str(ROOT / r.image), use_container_width=True)
                    if cols[i % 4].button("Selected" if ss["sample"] == r.sample_id else "Use", key=f"use_{r.sample_id}",
                                      use_container_width=True, disabled=ss["sample"] == r.sample_id):
                        use(r)
                        st.rerun()
                st.caption("All but the last image are patients the model never saw: first the showcase cases (selected because the model "
                           "detects the finding), then randomly picked ones. The last one was in the training population.")
                if ss.get("sample") in set(demo.sample_id):
                    rec = demo.set_index("sample_id").loc[ss["sample"]].to_dict()
                    image = ROOT / rec["image"]
            else:
                up = st.file_uploader("Frontal chest X-ray (JPG or PNG)", type=["jpg", "jpeg", "png"])
                if up is not None:
                    image = up
                    if ss.get("upload") != up.name:              # new upload: clear the previous patient's details
                        ss.update(upload=up.name, p_id="", p_note="")

            st.markdown("**Patient**")
            for k_, v_ in dict(p_id="", p_age=60, p_sex="Male", p_proj="PA", p_note="").items():
                ss.setdefault(k_, v_)
            p1, p2 = st.columns([2, 1])
            pid = p1.text_input("ID or name", key="p_id")
            age = p2.number_input("Age", 0, 110, key="p_age")
            p3, p4 = st.columns(2)
            sex = p3.selectbox("Sex", ["Male", "Female", "Unknown"], key="p_sex")
            proj = p4.selectbox("View", ["PA", "AP", "Other"], key="p_proj")
            note = st.text_input("Note", key="p_note")
            use_meta = st.checkbox("Give age, sex and view to the model", value=False,
                                   help="Off by default: it makes almost no difference to accuracy.")

        with out:
            if image is None:
                st.info("Choose an example or upload an image.")
            else:
                P = get_predictor(row.path)
                meta = dict(age=float(age), sex=sex, projection=proj) if use_meta else {}
                df, x = P.predict(np.array(Image.open(image).convert("L")), **meta)
                truth = {lb: rec.get(f"label_{lb}") for lb in LABELS} if rec else {}
                top = df.sort_values("prob", ascending=False)
                default = rec["finding"] if rec and rec["finding"] != "No Finding" else top.label.iloc[0]
                i1, i2 = st.columns(2)
                lab = i2.selectbox("Highlight regions for", LABELS, index=LABELS.index(default))
                cam, _ = P.cam(x, lab, **meta)
                fig, axs = plt.subplots(1, 2, figsize=(6, 3))
                for ax in axs:
                    ax.imshow(x[0, 0].cpu(), cmap="gray")
                    ax.axis("off")
                axs[1].imshow(cam, cmap="jet", alpha=0.4)
                fig.tight_layout()
                i1.pyplot(fig)
                plt.close(fig)
                i2.metric("Most likely", top.label.iloc[0], f"{top.prob.iloc[0]:.0%}", delta_color="off")

                d2 = df.assign(reliability=np.where(df.val_auroc >= 0.75, "good", np.where(df.val_auroc >= 0.65, "moderate", "weak")))
                if truth:
                    d2["label"] = [("● " if truth[lb] == 1 else "") + lb for lb in d2.label]
                bar = px.bar(d2.sort_values("prob"), x="prob", y="label", orientation="h", color="reliability", color_discrete_map=COLORS,
                             hover_data={"val_auroc": ":.3f"}, labels={"prob": "probability", "label": "", "reliability": "finding accuracy"})
                bar.update_xaxes(range=[0, 1], tickformat=".0%")
                bar.update_layout(height=430, margin=dict(t=10))
                st.plotly_chart(bar, use_container_width=True)
                st.caption(("● = present according to the CheXpert label. " if truth else "")
                           + "Colour shows how reliable the model is for that finding (VAL AUROC ≥ 0.75 good, ≥ 0.65 moderate).")

                if st.button("Save patient record", type="primary"):
                    DEMO.mkdir(exist_ok=True)
                    entry = dict(saved_at=datetime.now().isoformat(timespec="seconds"), image=getattr(image, "name", Path(str(image)).name),
                                 patient_id=pid, age=int(age), sex=sex, view=proj, note=note,
                                 model=f"{NAMES[row.tag]} ({row.scale}, seed {row.seed})", metadata_used=use_meta,
                                 top3=", ".join(f"{r.label} {r.prob:.0%}" for r in top.head(3).itertuples()),
                                 probabilities={r.label: round(float(r.prob), 4) for r in df.itertuples()},
                                 chexpert_positive=[lb for lb, v in truth.items() if v == 1] if truth else None)
                    with open(RECORDS, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(entry) + "\n")
                    st.success(f"Saved to {RECORDS.relative_to(ROOT).as_posix()}")

        if RECORDS.exists():
            with st.expander("Saved records"):
                recs = pd.DataFrame([json.loads(s) for s in RECORDS.read_text(encoding="utf-8").splitlines() if s.strip()])
                st.dataframe(recs.drop(columns=["probabilities"], errors="ignore"), use_container_width=True, hide_index=True)
                st.download_button("Download", RECORDS.read_bytes(), "patient_records.jsonl")
        st.caption("Research prototype. Not for clinical use.")

# ------------------------------------------------------------------ privacy
with tab_priv:
    st.markdown("A **membership-inference attack** guesses whether a patient was in the training data, using the fact that models "
                "usually fit their training data better. An attack score (AUC) of **0.5 means it does no better than a coin flip**.")
    mia_dir = ROOT / "results" / "mia_demo"
    st.subheader("Overfit model, with and without DP")
    if (mia_dir / "summary.csv").exists():
        S = pd.read_csv(mia_dir / "summary.csv")
        mj = json.loads((mia_dir / "meta.json").read_text())
        g = S.groupby("arm", sort=False).agg(auc=("auc", "mean"), auc_sd=("auc", "std"), val_auroc=("val_auroc", "mean"),
                                             member_loss=("member_loss", "mean"), heldout_loss=("nonmember_loss", "mean"),
                                             eps=("eps", "mean")).reset_index()
        for col, r in zip(st.columns(len(g)), g.itertuples()):
            col.metric(r.arm.replace("eps", "ε"), f"{r.auc:.2f}", f"accuracy {r.val_auroc:.2f}", delta_color="off")
        h1, h2 = st.columns(2)
        fa = px.bar(g.assign(arm=g.arm.str.replace("eps", "ε")), x="arm", y="auc", error_y="auc_sd",
                    labels={"auc": "attack AUC", "arm": ""})
        fa.add_hline(y=0.5, line_dash="dot", annotation_text="coin flip")
        fa.update_yaxes(range=[0.4, 1.0]).update_layout(height=340)
        h1.plotly_chart(fa, use_container_width=True)
        arm = h2.selectbox("Loss of training vs unseen patients", list(g.arm), format_func=lambda a: a.replace("eps", "ε"))
        Ld = pd.read_csv(mia_dir / "losses_seed0.csv").query("arm == @arm")
        Ld["patients"] = np.where(Ld.member, "in training", "unseen")
        fh = px.histogram(Ld, x="loss", color="patients", barmode="overlay", opacity=0.6, nbins=40, histnorm="probability density",
                          range_x=[0, min(6.0, float(Ld.loss.quantile(0.99)))])
        fh.update_layout(height=300)
        h2.plotly_chart(fh, use_container_width=True)
        st.caption(f"{mj['members']} training patients, {mj['steps']} steps, {mj['seeds']} seeds. The model without DP is trained to overfit on "
                   "purpose. With DP the two loss distributions overlap and the attack fails. At this small size DP costs accuracy, "
                   "and ε=1 costs a lot.")
    else:
        st.info("Run `python scripts/mia_demo.py` to create this panel.")

    lira_dir = ROOT / "results" / "mia_lira"
    if (lira_dir / "stress" / "summary.csv").exists():
        st.subheader("Stronger attack: likelihood ratio with shadow models")
        st.markdown("The attacker trains many copies of the model (shadow models), with and without each patient, and learns how that "
                    "patient's prediction changes when they are in the training data. This is the strongest standard attack (LiRA). "
                    "What matters for privacy is how many training patients it can identify while almost never "
                    "accusing an unseen patient: the **true-positive rate at a 1% false-positive rate**.")
        L = pd.read_csv(lira_dir / "stress" / "summary.csv")
        best = L[L.attack == "LiRA online"].reset_index(drop=True)
        for col, (_, r) in zip(st.columns(len(best)), best.iterrows()):
            col.metric(r.arm.replace("eps", "ε"), f"{100 * r['tpr@0.01']:.1f}% found", f"AUC {r.auc:.2f}", delta_color="off")
        st.caption("Share of training patients identified at a 1% false-positive rate (chance = 1%).")
        g1, g2 = st.columns(2)
        R = pd.read_csv(lira_dir / "stress" / "roc.csv").query("attack == 'LiRA online'")
        fr = px.line(R.assign(arm=R.arm.str.replace("eps", "ε")), x="fpr", y="tpr", color="arm", log_x=True, log_y=True,
                     labels={"fpr": "false-positive rate", "tpr": "true-positive rate", "arm": ""})
        fr.add_scatter(x=[1e-4, 1], y=[1e-4, 1], mode="lines", line=dict(dash="dot", color="gray"), name="chance")
        fr.update_xaxes(range=[-3, 0]).update_yaxes(range=[-3, 0]).update_layout(height=360)
        g1.plotly_chart(fr, use_container_width=True)
        dp = L[L.eps_claimed.notna()].groupby("arm", sort=False).agg(claimed=("eps_claimed", "first"),
                                                                     measured=("eps_lower_bound", "max")).reset_index()
        if len(dp):
            fe = px.bar(dp.melt("arm", var_name="ε", value_name="value").assign(arm=lambda d: d.arm.str.replace("eps", "ε")),
                        x="arm", y="value", color="ε", barmode="group", labels={"value": "ε", "arm": ""})
            fe.update_layout(height=360)
            g2.plotly_chart(fe, use_container_width=True)
            g2.caption("Audit: the ε an attack demonstrates (lower bound) must stay below the ε the accountant guarantees.")
        if (lira_dir / "real" / "summary.csv").exists():
            Rr = pd.read_csv(lira_dir / "real" / "summary.csv")
            Rr["model"] = Rr.target.map(lambda t: NAMES.get(t, t))
            st.markdown("**The full-scale models** (2,000 training vs 2,000 unseen patients, reference models trained without them):")
            st.dataframe(Rr[["model", "attack", "auc", "tpr@0.01", "tpr@0.001"]].rename(
                columns={"auc": "AUC", "tpr@0.01": "found at 1% FPR", "tpr@0.001": "found at 0.1% FPR"}).round(3),
                use_container_width=True, hide_index=True)

    st.subheader("Attack the trained models")
    st.caption("Training patients vs unseen VAL patients, 1,000 of each.")
    if len(rt):
        a, b = st.columns(2)
        sc = a.selectbox("Trained on", scale_options(rt), key="mia_scale")
        r1 = rt[(rt.scale == sc) & (rt.seed == 0)]
        opts = [x for x in ("E3", "E1", "E4_eps8", "E4_eps3", "E4_eps1", "E5_eps8") if x in set(r1.tag)]
        picks = b.multiselect("Models", opts, default=[x for x in ("E3", "E4_eps3") if x in opts], format_func=NAMES.get)
        if st.button("Run attack") and picks:
            res = []
            for tg in picks:
                with st.spinner(f"Attacking {NAMES[tg]}…"):
                    from dpmmfl.serve import live_mia
                    res.append((tg, live_mia(r1[r1.tag == tg].iloc[0].path)))
            for col, (tg, r) in zip(st.columns(len(res)), res):
                col.metric(NAMES[tg], f"AUC {r['auc']:.3f}", f"{100 * r['tpr1']:.1f}% found at 1% FPR", delta_color="off")
                D = pd.DataFrame({"loss": np.r_[r["member_loss"], r["nonmember_loss"]],
                                  "patients": ["in training"] * r["n"] + ["unseen"] * r["n"]})
                col.plotly_chart(px.histogram(D, x="loss", color="patients", barmode="overlay", opacity=0.6, nbins=40,
                                              range_x=[0, 4]).update_layout(height=280), use_container_width=True)
            st.caption("With thousands of training patients and early stopping, even the model without DP leaks very little. "
                       "DP adds a guarantee that holds for any attack, including ones stronger than this.")

# ------------------------------------------------------------------ about
with tab_about:
    st.markdown("""
**Model.** A frozen DenseNet-121 (TorchXRayVision, trained on NIH data, never on CheXpert) turns each X-ray into a feature vector;
a small classifier on top predicts 14 findings. Only the classifier is trained.

**Federated learning.** Five simulated hospitals with different patient mixes train locally and share only model updates (FedAvg).

**Differential privacy.** Each patient's contribution to every update is clipped and Gaussian noise is added (DP-SGD), so the final model
reveals provably little about any single patient. ε is the privacy budget: smaller is stronger.

**Communication.** Updates are sent as 8-bit integers (up) and 16-bit floats (down), measured on the wire, with no loss in accuracy.

**Fairness.** Accuracy is checked across sex, age and X-ray view. Per-group decision thresholds reduce the sensitivity gap between views.

**Limits.**
- TEST has 202 images, so differences under about 0.03 AUROC are noise.
- The privacy guarantee covers the trained classifier, not the public pre-trained encoder. Hyper-parameter tuning on VAL is not privacy-accounted.
- Federation is simulated on one machine.
- Some findings (lung lesion, pneumonia, enlarged cardiomediastinum) are close to chance.
- Saved patient records stay in `demo/patient_records.jsonl` on this machine. Use de-identified data.
- Research prototype, not a medical device.
""")
