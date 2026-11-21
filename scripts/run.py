"""Run one experiment for one seed.

  python scripts/run.py --exp E4 --eps 3 --seed 0 [--final] [--resume] [--smoke] [--config f.yaml] [--set k=v] [--fair F1] [--tag x]

E0 linear probe | E1 central image-only | E2 central (cfg head, default FiLM) | E3 FedAvg | E4 patient-level DP-FedAvg
E5 image-level DP | E7 E4 + fairness F1 (group reweighting) | E8 FL with compressed traffic (int8 up / fp16 down unless --set comm.*;
DP when --eps is given). Every run also writes the E9 membership-inference audit (mia.json) and the F2 group-threshold report.
TEST is only evaluated with --final (never in --smoke), once per run_id, and logged to <results>/test_access.jsonl.
"""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dpmmfl.cli import parser, config
from dpmmfl.constants import LABELS
from dpmmfl.data.store import ClientData, load_split, pos_weight_from
from dpmmfl.eval.fairness import subgroup_audit, client_audit, group_weights, group_thresholds
from dpmmfl.eval.metrics import evaluate, youden_thresholds, fit_temperature
from dpmmfl.eval.mia import mia_audit
from dpmmfl.models.head import build_head
from dpmmfl.results import RunWriter, RunExists, TestGuard
from dpmmfl.train.central import train_central
from dpmmfl.train.infer import predict_logits, sigmoid
from dpmmfl.train.losses import make_loss
from dpmmfl.train.server import run_fl
from dpmmfl.utils import seed_everything, get_device, get_logger

EXPS = {"E0": dict(kind="central", head="linear"),
        "E1": dict(kind="central", head="image_only"),
        "E2": dict(kind="central"),
        "E3": dict(kind="fl", dp=False),
        "E4": dict(kind="fl", dp=True, unit="patient"),
        "E5": dict(kind="fl", dp=True, unit="image"),
        "E7": dict(kind="fl", dp=True, unit="patient", fair="F1"),
        "E8": dict(kind="fl", dp=None, unit="patient", comm=("int8", "fp16"))}


def rows_of(meta, K):
    pool = (meta["split"] == "pool").to_numpy()
    lv = meta["local_val"].to_numpy()
    train = pool & ~lv
    cid = meta["client_id"].to_numpy()
    return (np.flatnonzero((meta["split"] == "val").to_numpy()), np.flatnonzero(train),
            [np.flatnonzero(train & (cid == k)) for k in range(K)],
            {k: np.flatnonzero(pool & lv & (cid == k)) for k in range(K)})


def resolve(a, cfg):
    e = EXPS[a.exp]
    if e.get("head"):
        cfg["model"]["head"] = e["head"]
    if e.get("comm") and not any(s.startswith("comm.") for s in a.set):
        cfg["comm"]["up"], cfg["comm"]["down"] = e["comm"]
    cfg["fair"]["method"] = a.fair or e.get("fair") or cfg["fair"]["method"]
    dp_on = e["dp"] if e.get("dp") is not None else a.eps is not None
    eps = a.eps if a.eps is not None else cfg["dp"]["target_eps"]
    unit = e.get("unit", "patient")
    if e["kind"] == "central":
        dp_on = False
    tag = a.exp + (f"_eps{eps:g}" if dp_on else "") + (f"_{cfg['fair']['method']}" if cfg["fair"]["method"] != "none" else "")
    if e["kind"] == "fl" and (cfg["comm"]["up"], cfg["comm"]["down"]) != ("fp32", "fp32"):
        tag += f"_up{cfg['comm']['up']}_dn{cfg['comm']['down']}"
    tag += (f"_{a.tag}" if a.tag else "")
    cfg["run"] = {"exp": a.exp, "seed": a.seed, "eps": eps if dp_on else None, "dp": dp_on, "unit": unit, "final": a.final}
    return e, dp_on, eps, unit, tag


def main():
    ap = parser(__doc__)
    ap.add_argument("--exp", required=True, choices=EXPS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--fair", choices=["none", "F1"], default=None, help="F1 = group-reweighted loss (weights <= 1)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--final", action="store_true", help="evaluate TEST once (final configs only)")
    ap.add_argument("--resume", action="store_true", help="skip if complete; redo if partial")
    a = ap.parse_args()
    cfg = config(a)
    e, dp_on, eps, unit, tag = resolve(a, cfg)
    try:
        W = RunWriter(cfg, tag, a.seed, resume=a.resume)
    except RunExists as ex:
        print(f"REFUSED: {ex}"); sys.exit(3)
    if W.done:
        print(f"SKIP (complete): {W.run_id}"); return
    guard = TestGuard(cfg, W, a.final)
    if a.final:
        guard.check()                                     # fail fast, before hours of training
    W.start()
    log = get_logger(file=W.dir / "run.log").info
    seed_everything(a.seed)
    dev = get_device(cfg)
    K = cfg["partition"]["K"]

    train = load_split(cfg, "train", dev)
    meta = train.meta
    val_rows, train_rows, client_rows, lv_rows = rows_of(meta, K)
    pw = pos_weight_from(train, train_rows)
    loss_fn = make_loss(cfg, pw)
    if cfg["fair"]["method"] == "F1":
        fw, sizes = group_weights(meta, train_rows, cfg["fair"]["attrs"], cfg["fair"]["wmin"])
        train.w[torch.as_tensor(train_rows, device=dev)] = torch.as_tensor(fw, device=dev)
        log(f"F1 reweighting on {cfg['fair']['attrs']}: {len(sizes)} groups, w in [{fw.min():.2f}, {fw.max():.2f}] (<= 1)")
    model = build_head(cfg).to(dev)
    log(f"{W.run_id} dev={dev} features={cfg['features']['encoders']} on_gpu={train.on_gpu} head={cfg['model']['head']} "
        f"loss={cfg['model']['loss']} params={sum(p.numel() for p in model.parameters())} train={len(train_rows)} val={len(val_rows)}")
    if e["kind"] == "central":
        best, hist, info = train_central(cfg, model, train_rows, val_rows, train, loss_fn, a.seed, log)
    else:
        clients = [ClientData(train, r) for r in client_rows]
        best, hist, info = run_fl(cfg, model, clients, val_rows, train, loss_fn, a.seed, log, dp_on=dp_on, unit=unit, eps=eps)
        if dp_on and unit == "image":
            kmax = max(c.max_images for c in clients)
            info["image_level_worst_patient_eps_bound"] = kmax * eps          # group privacy (loose)
    model.load_state_dict(best)

    # ---- VAL: temperature, thresholds, metrics, fairness (F2 always reported), MIA
    mv_meta = meta.iloc[val_rows].reset_index(drop=True)
    zv = predict_logits(model, train, val_rows)
    yv, mv = train.numpy_targets(val_rows)
    temps = fit_temperature(zv, yv, mv)
    pv = sigmoid(zv / temps)
    thr = youden_thresholds(yv, pv, mv)
    ev = cfg["eval"]
    res_val = evaluate(yv, pv, mv, mv_meta["patient_id"].to_numpy(), ev["min_pos_val"], n_boot=ev["bootstrap_val"], thr=thr)
    res_val["info"] = info
    W.json("metrics_val.json", res_val)
    W.preds("preds_val.parquet", mv_meta, zv, temps)
    f2, thr_g = group_thresholds(mv_meta, yv, pv, mv, cfg["fair"]["thr_attr"], seed=a.seed)
    W.json("fairness.json", {
        "method": cfg["fair"]["method"],
        "subgroups_val": subgroup_audit(mv_meta, yv, pv, mv, thr, ev["bootstrap_val"], ev["min_pos_group"]),
        "clients_localval": client_audit(meta, lv_rows, lambda r: sigmoid(predict_logits(model, train, r) / temps),
                                         train.numpy_targets, thr, ev["bootstrap_val"], ev["min_pos_group"]),
        "F2_group_thresholds": f2, "F2_thresholds": thr_g})
    W.json("mia.json", mia_audit(model, train, train_rows, val_rows, loss_fn, a.seed))
    W.json("calibration.json", {"temperature": dict(zip(LABELS, temps.tolist())), "thresholds": dict(zip(LABELS, thr.tolist()))})
    if e["kind"] == "fl":
        up, dn, base = info["comm_bytes_up"], info["comm_bytes_down"], info["comm_bytes_fp32_baseline"]
        W.json("comms.json", {"n_params": sum(p.numel() for p in model.parameters()), "mode_up": cfg["comm"]["up"],
                              "mode_down": cfg["comm"]["down"], "rounds_run": info["rounds_run"], "MB_up": up / 1e6,
                              "MB_down": dn / 1e6, "MB_total": (up + dn) / 1e6, "MB_fp32_same_rounds": base / 1e6,
                              "saving_vs_fp32_pct": 100 * (1 - (up + dn) / base)})
    if dp_on:
        W.json("privacy.json", {k: info[k] for k in info if k.startswith(("eps", "sigma", "q", "delta", "planned", "actual", "accountant", "image_level"))})

    # ---- TEST (final runs only, once)
    extra = {"val_mean5": res_val["mean5_auroc"], "eps_spent_prv": info.get("eps_spent_max_prv")}
    if a.final:
        guard.check()
        test = load_split(cfg, "test", dev)
        zt = predict_logits(model, test)
        guard.record()
        yt, mt = test.numpy_targets(np.arange(len(test)))
        pt = sigmoid(zt / temps)
        res_test = evaluate(yt, pt, mt, test.meta["patient_id"].to_numpy(), ev["min_pos_test"], n_boot=ev["bootstrap"], thr=thr)
        W.json("metrics_test.json", res_test)
        W.preds("preds_test.parquet", test.meta, zt, temps)
        extra["test_mean5"] = res_test["mean5_auroc"]
    hist.to_csv(W.dir / "history.csv", index=False)
    torch.save(model.state_dict(), W.dir / "head.pt")
    W.finish(extra)
    log(f"DONE {W.run_id}: VAL mean5 {res_val['mean5_auroc']:.4f} CI {np.round(res_val.get('mean5_auroc_ci95', [np.nan] * 2), 3).tolist()}"
        + (f" | eps PRV {info['eps_spent_max_prv']:.3f} RDP {info['eps_spent_max_rdp']:.3f}" if dp_on else "")
        + (f" | TEST {extra['test_mean5']:.4f}" if a.final else "") + f" | {W.manifest['wall_sec']}s")


if __name__ == "__main__":
    main()
