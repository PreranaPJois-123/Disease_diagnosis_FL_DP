"""FedAvg / FedAvgM (+ server EMA) round loop with optional per-client DP and compressed traffic (docs/05, 06)."""
import copy, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from .client import local_train
from .comm import ErrorFeedback, transmit
from .infer import EarlyStopper, val_score
from .privacy import calibrate_sigma, ClientAccountant


def load_vec(model, vec):
    """Copy a flat vector into model parameters. vector_to_parameters alone makes params *views* of vec, so
    in-place optimizer steps would silently mutate the server's weights (clients would train sequentially)."""
    vector_to_parameters(vec.detach().clone(), model.parameters())


def setup_dp(cfg, clients, eps, unit, log):
    """Per-client σ so every client reaches the same target ε (PRV) after the planned R×S steps."""
    fl, dpc = cfg["fl"], cfg["dp"]
    steps = fl["rounds"] * fl["local_steps"]
    acc = dpc.get("accountant", "prv")
    cache_f = Path(cfg["cache_dir"]) / "sigma_cache.json"          # σ depends only on (ε, δ, q, steps, accountant)
    cache = json.loads(cache_f.read_text()) if cache_f.exists() else {}
    out = []
    for k, c in enumerate(clients):
        units = c.n_images if unit == "image" else c.n_patients
        q = min(1.0, dpc["batch"] / units)
        key = f"{acc}|{eps:g}|{dpc['delta']:g}|{q:.10f}|{steps}"
        if key not in cache:
            cache[key] = calibrate_sigma(eps, dpc["delta"], q, steps, acc)
            cache_f.write_text(json.dumps(cache, indent=1))
        sigma = cache[key]
        out.append(dict(sigma=sigma, q=q, unit=unit, M=1 if unit == "image" else dpc["max_images"], C=dpc["clip"],
                        B=dpc["batch"], lr=dpc["lr"], acct=ClientAccountant(sigma, q, dpc["delta"])))
        log(f"client {k}: units={units} q={q:.4f} sigma={sigma:.3f} planned_steps={steps}")
    return out


def run_fl(cfg, model, clients, val_rows, store, loss_fn, seed, log, dp_on=False, unit="patient", eps=None):
    """clients: list[ClientData]. Returns (best state_dict, history DataFrame, info dict)."""
    fl, cc = cfg["fl"], cfg["comm"]
    R, S = fl["rounds"], fl["local_steps"]
    server = cfg["dp"]["server"] if dp_on else fl["server"]
    p_k = np.array([c.n_patients for c in clients], float)
    p_k /= p_k.sum()
    dps = setup_dp(cfg, clients, eps, unit, log) if dp_on else [None] * len(clients)
    dev = store.device
    w = parameters_to_vector(model.parameters()).detach().clone()
    mom, w_ema = torch.zeros_like(w), w.clone()
    ema = fl.get("server_ema", 0.0)
    ef = ErrorFeedback()
    stopper = EarlyStopper(fl.get("patience", 0))
    hist, bytes_up, bytes_down, t0 = [], 0, 0, time.time()
    min_pos, every = cfg["eval"]["min_pos_val"], max(1, fl.get("eval_every", 1))
    rounds_run = 0
    for r in range(1, R + 1):
        rounds_run = r
        agg = torch.zeros_like(w)
        w_dn, nb_dn = transmit(w, cc["down"], cc, dev)           # one broadcast payload, sent to every client
        clipped = []
        for k, c in enumerate(clients):
            local = copy.deepcopy(model)
            load_vec(local, w_dn)
            gen = torch.Generator(device=dev).manual_seed(int(seed) * 1_000_003 + r * 1009 + k)
            delta, frac = local_train(local, c, cfg, S, loss_fn, gen, dps[k])
            sent, nb_up = ef.compress(k, delta, cc["up"], cc)      # dequantised payload the server actually receives
            agg += p_k[k] * sent
            bytes_up += nb_up
            bytes_down += nb_dn
            if frac is not None:
                clipped.append(frac)
        if server == "fedavgm":
            mom = fl["beta"] * mom + agg
            w = w + fl["server_lr"] * mom
        else:
            w = w + fl["server_lr"] * agg
        w_ema = w if ema <= 0 else ema * w_ema + (1 - ema) * w
        if r % every and r != R:
            continue
        load_vec(model, w_ema)
        auc = val_score(model, store, val_rows, min_pos)
        eps_k = [d["acct"].epsilon_rdp() for d in dps] if dp_on else []
        hist.append({"round": r, "val_mean5_auroc": auc, "eps_rdp_max": max(eps_k) if eps_k else 0.0,
                     "clipped_frac": float(np.mean(clipped)) if clipped else np.nan, "sec": time.time() - t0,
                     **{f"eps_rdp_c{k}": e for k, e in enumerate(eps_k)}})
        log(f"round {r:3d} val mean5 AUROC {auc:.4f}" + (f" eps_rdp_max {max(eps_k):.2f} clipped {np.mean(clipped):.2f}" if eps_k else ""))
        if stopper.update(auc, r, model):
            log(f"early stop at round {r} (best round {stopper.best_step})")
            break
    info = {"best_round": stopper.best_step, "best_val_mean5": stopper.best, "rounds_run": rounds_run,
            "planned_steps": R * S, "actual_steps": rounds_run * S,
            "comm_bytes_up": bytes_up, "comm_bytes_down": bytes_down,
            "comm_bytes_fp32_baseline": 2 * len(clients) * rounds_run * (w.numel() * 4)}
    if dp_on:
        eps_spent = [d["acct"].epsilon() for d in dps]               # after the steps actually taken (conservative)
        info.update({"sigmas": [d["sigma"] for d in dps], "q": [d["q"] for d in dps],
                     "eps_target": eps, "delta": cfg["dp"]["delta"], "accountant_calibration": cfg["dp"].get("accountant", "prv"),
                     "eps_spent_prv": [e[0] for e in eps_spent], "eps_spent_rdp": [e[1] for e in eps_spent],
                     "eps_spent_max_prv": max(e[0] for e in eps_spent), "eps_spent_max_rdp": max(e[1] for e in eps_spent)})
    return stopper.state, pd.DataFrame(hist), info
