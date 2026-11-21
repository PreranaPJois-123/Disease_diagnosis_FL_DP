"""Fast unit tests (< 1 min, CPU only)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dpmmfl.constants import LABELS, N_LABELS
from dpmmfl.data.parse import make_targets
from dpmmfl.data.splits import assign_splits, assert_disjoint
from dpmmfl.models.head import FusionHead
from dpmmfl.train.losses import masked_bce_per_image, masked_asl_per_image
from dpmmfl.train.privacy import unit_grads, clip_and_sum, dp_gradient, calibrate_sigma, ClientAccountant, epsilon_for

E = 2   # encoders in the synthetic batches


def test_label_mapping():
    raw = {l: [np.nan, -1.0, 1.0, 0.0] for l in LABELS}
    y, m = make_targets(pd.DataFrame(raw), {"Atelectasis": "ones", "Cardiomegaly": "zeros"}, True)
    a, c, o = LABELS.index("Atelectasis"), LABELS.index("Cardiomegaly"), LABELS.index("Fracture")
    assert y[:, a].tolist() == [0, 1, 1, 0] and m[:, a].all()
    assert y[:, c].tolist() == [0, 0, 1, 0] and m[:, c].all()
    assert y[:, o].tolist() == [0, 0, 1, 0] and m[:, o].tolist() == [1, 0, 1, 1]      # uncertain -> ignored


def test_split_overlap_zero():
    df = pd.DataFrame({"patient_id": [f"patient{i % 50:05d}" for i in range(300)]})
    d = assign_splits(df, 0.1, 42)
    assert_disjoint({"val": set(d[d.split == "val"].patient_id), "pool": set(d[d.split == "pool"].patient_id)})
    with pytest.raises(AssertionError):
        assert_disjoint({"a": {"p1", "p2"}, "b": {"p2"}})


def _batch(B=12, M=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    F = torch.randn(B, M, E, 1024, generator=g); T = torch.randn(B, M, 9, generator=g)
    Y = (torch.rand(B, M, N_LABELS, generator=g) > 0.7).float(); Mk = torch.ones(B, M, N_LABELS)
    IM = (torch.rand(B, M, generator=g) > 0.3).float(); IM[:, 0] = 1
    return F, T, Y, Mk, IM


def _model(head="film"):
    torch.manual_seed(0)
    m = FusionHead(E, head); m.eval()
    if head == "film":
        for p in m.tab[-1].parameters():
            torch.nn.init.normal_(p, std=0.02)                       # non-zero so FiLM params get grads
    return m


LOSS = lambda z, y, m: masked_bce_per_image(z, y, m, torch.ones(N_LABELS))


def test_per_patient_norm_bounded_by_C():
    g = unit_grads(_model(), LOSS, *_batch())
    C = 0.05
    _, norms = clip_and_sum(g, C)
    assert norms.max() > C                                            # clipping is active
    f = (C / (norms + 1e-6)).clamp(max=1)
    after = torch.sqrt(sum((x * f.view(-1, *[1] * (x.dim() - 1))).flatten(1).pow(2).sum(1) for x in g.values()))
    assert after.max() <= C + 1e-5


@pytest.mark.parametrize("head", ["film", "linear"])
def test_dp_equals_plain_gradient_when_sigma0_C_inf(head):
    m = _model(head)
    F, T, Y, Mk, IM = _batch()
    W = torch.rand_like(IM) * 0.5 + 0.5
    B = F.shape[0]
    dp_gradient(m, LOSS, (F, T, Y, Mk, IM, W), C=float("inf"), sigma=0.0, B_expected=B)
    got = {k: p.grad.clone() for k, p in m.named_parameters()}
    m.zero_grad()
    tot = sum((LOSS(m(F[i], T[i]), Y[i], Mk[i]) * W[i] * IM[i]).sum() / IM[i].sum() for i in range(B))
    (tot / B).backward()
    for k, p in m.named_parameters():
        assert torch.allclose(got[k], p.grad, atol=1e-5), k


def test_noise_scale():
    m = _model()
    dp_gradient(m, LOSS, None, C=1.0, sigma=2.0, B_expected=100)      # empty sample -> noise only
    g = torch.cat([p.grad.flatten() for p in m.parameters()])
    assert abs(g.std().item() - 0.02) < 0.002


def test_prv_eps_le_rdp_eps():
    for q, steps in ((0.05, 600), (0.01, 1500)):
        prv, rdp = epsilon_for(1.2, q, steps, 1e-5)
        assert 0 < prv <= rdp


def test_equal_eps_across_clients():
    steps, delta, target = 300, 1e-5, 3.0
    eps = []
    for q in (0.02, 0.2):
        a = ClientAccountant(calibrate_sigma(target, delta, q, steps, "prv"), q, delta)
        a.step(steps)
        eps.append(a.epsilon()[0])
    assert max(eps) <= target + 0.02 and max(eps) - min(eps) < 0.05, eps


def test_compression_roundtrip_and_aggregate_changes():
    from dpmmfl.train.comm import encode, decode, serialize, deserialize, transmit, ErrorFeedback
    g = torch.Generator().manual_seed(0)
    deltas = [torch.randn(10_001, generator=g) * 0.01 for _ in range(3)]
    n = deltas[0].numel()
    # fp32 is exact; bytes are the real serialised length
    r32, b32 = transmit(deltas[0], "fp32", {}, "cpu")
    assert torch.equal(r32, deltas[0]) and b32 == len(serialize(encode(deltas[0], "fp32"))) and b32 >= 4 * n
    # int8 is lossy (but bounded) and ~4x smaller
    r8, b8 = transmit(deltas[0], "int8", {"block": 1024}, "cpu")
    err = (deltas[0] - r8).abs().max()
    assert err > 0 and err <= deltas[0].abs().max() / 127 / 2 + 1e-7
    assert b8 < 0.3 * b32
    p = deserialize(serialize(encode(deltas[0], "int8", 1024)))
    assert torch.equal(decode(p, "int8"), r8)
    # the aggregate of dequantised updates differs from the fp32 aggregate
    agg32 = sum(transmit(d, "fp32", {}, "cpu")[0] for d in deltas) / 3
    agg8 = sum(transmit(d, "int8", {"block": 1024}, "cpu")[0] for d in deltas) / 3
    assert not torch.equal(agg32, agg8) and (agg32 - agg8).abs().max() < 1e-3
    # top-k + error feedback: residual = (update + residual) - sent, so nothing is lost over time
    ef, cc = ErrorFeedback(), {"topk": 0.1}
    sent, bk = ef.compress(0, deltas[0], "topk", cc)
    assert (sent != 0).sum() == int(np.ceil(0.1 * n)) and bk < 0.2 * b32
    assert torch.allclose(sent + ef.res[0], deltas[0])


def test_results_writer_refuses_overwrite(tmp_path):
    from dpmmfl.results import RunWriter, RunExists, TestGuard, TestAccessDenied
    cfg = {"results_dir": str(tmp_path), "features": {"encoders": ["nih"]}, "smoke": False, "x": 1}
    w = RunWriter(cfg, "E3", 0)
    w.start(); w.finish()
    with pytest.raises(RunExists):
        RunWriter(cfg, "E3", 0)
    assert RunWriter(cfg, "E3", 0, resume=True).done                     # --resume skips a complete run
    g = TestGuard(cfg, w, final=True)
    g.check(); g.record()
    with pytest.raises(TestAccessDenied):
        g.check()                                                        # TEST once per run_id
    with pytest.raises(TestAccessDenied):
        TestGuard({**cfg, "smoke": True}, w, final=True).check()         # never in smoke mode


def test_load_vec_does_not_alias():
    from dpmmfl.train.server import load_vec
    m = _model()
    vec = torch.cat([p.detach().flatten() for p in m.parameters()]).clone()
    ref = vec.clone()
    load_vec(m, vec)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
    assert torch.equal(vec, ref)


def test_group_weights_le_one():
    from dpmmfl.eval.fairness import group_weights
    rng = np.random.default_rng(0)
    meta = pd.DataFrame({"Age": rng.integers(20, 90, 500), "AP/PA": rng.choice(["AP", "PA"], 500, p=[.9, .1]),
                         "Sex": rng.choice(["Male", "Female"], 500)})
    w, sizes = group_weights(meta, np.arange(500))
    assert w.max() <= 1.0 and w.min() > 0 and len(sizes) > 1
    pa = (meta["AP/PA"] == "PA").to_numpy()
    assert w[pa].mean() > w[~pa].mean()                                   # rarer group up-weighted


def test_asl_finite_and_masked():
    z = torch.randn(4, N_LABELS) * 5
    y = (torch.rand(4, N_LABELS) > 0.5).float()
    m = torch.ones_like(y); m[:, 3] = 0
    l = masked_asl_per_image(z, y, m)
    assert torch.isfinite(l).all() and (l >= 0).all()
    z2 = z.clone(); z2[:, 3] = 100
    assert torch.allclose(l, masked_asl_per_image(z2, y, m))              # masked label has no effect


def test_dashboard_scale_names():
    from dpmmfl.serve import ROOT, _scale_of
    r = ROOT / "results"
    assert _scale_of(r / "scale_small" / "runs" / "x") == "small"
    assert _scale_of(r / "scale_medium" / "runs" / "x") == "medium"
    assert _scale_of(r / "v2.1-final" / "runs" / "x") == "large (v2.1)"
    assert _scale_of(r / "v2-large" / "E4_eps3" / "0") == "large (v2)"


def test_fast_auroc_matches_sklearn_and_bootstrap_weights():
    from sklearn.metrics import roc_auc_score
    from dpmmfl.eval.metrics import safe_auroc, weighted_auroc
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.3).astype(float)
    s = np.round(rng.normal(size=400) + y, 1)                              # rounded: many ties
    assert abs(safe_auroc(y, s) - roc_auc_score(y, s)) < 1e-12
    W = rng.integers(0, 3, size=(5, 400)).astype(float)                    # integer multiplicities = a resample
    for w, a in zip(W, weighted_auroc(y, s, W)):
        idx = np.repeat(np.arange(400), w.astype(int))
        assert abs(a - roc_auc_score(y[idx], s[idx])) < 1e-12
    assert np.isnan(weighted_auroc(np.ones(4), s[:4], np.ones((1, 4))))[0]   # one class only -> undefined



def test_lira_detects_shift_and_dp_bounds():
    from dpmmfl.eval.lira import dp_roc_bound, eps_lower_bound, lira_offline, lira_online, roc_summary
    rng = np.random.default_rng(0)
    K, P = 32, 2000
    base = rng.normal(size=P)                                              # per-patient difficulty
    IN = rng.random((K, P)) < 0.5
    for shift, lo, hi in ((1.0, 0.85, 1.0), (0.0, 0.45, 0.55)):
        S = base + shift * IN + 0.3 * rng.normal(size=(K, P))
        sc_on = np.concatenate([lira_online(S[k], S[np.arange(K) != k], IN[np.arange(K) != k]) for k in range(4)])
        sc_off = np.concatenate([lira_offline(S[k], S[np.arange(K) != k], ~IN[np.arange(K) != k]) for k in range(4)])
        lab = IN[:4].ravel()
        for sc in (sc_on, sc_off):
            assert lo < roc_summary(sc, lab)[0]["auc"] < hi
        lb = eps_lower_bound(sc_on, lab, 1e-5)
        assert (lb > 1.0) if shift else (lb < 0.3)
    f = np.linspace(0, 1, 101)
    assert np.allclose(dp_roc_bound(0.0, 0.0, f), f)                       # eps = 0: no attack beats chance
    assert (dp_roc_bound(3.0, 1e-5, f) >= dp_roc_bound(1.0, 1e-5, f)).all()
