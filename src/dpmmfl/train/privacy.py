"""Patient-level DP-SGD (docs/06). Per-unit grads via torch.func vmap(grad); accounting with Opacus PRV (primary) and RDP (reported).

The released ε is the PRV ε after the steps actually taken; RDP is an upper bound reported alongside it.
"""
import torch
from torch.func import functional_call, vmap, grad


_SIGMA_CACHE = {}


def calibrate_sigma(eps, delta, q, steps, accountant="prv", tol=0.005):
    """Smallest σ with ε(σ) <= eps. RDP via Opacus; PRV by bisection inside [0.5 σ_RDP, σ_RDP]
    (PRV is tighter, so σ_PRV <= σ_RDP), which needs ~10 PRV evaluations instead of Opacus' ~25."""
    key = (round(eps, 6), delta, round(q, 8), steps, accountant)
    if key in _SIGMA_CACHE:
        return _SIGMA_CACHE[key]
    from opacus.accountants.utils import get_noise_multiplier
    hi = float(get_noise_multiplier(target_epsilon=eps, target_delta=delta, sample_rate=q, steps=steps,
                                    accountant="rdp", epsilon_tolerance=tol))
    if accountant == "prv":
        prv = lambda s: epsilon_for(s, q, steps, delta)[0]
        lo = 0.5 * hi
        while prv(lo) <= eps:                       # make sure the bracket holds the target
            hi, lo = lo, lo * 0.5
        while True:
            mid = 0.5 * (lo + hi)
            e = prv(mid)
            if e > eps:
                lo = mid
            else:
                hi = mid
                if eps - e < tol:
                    break
            if hi - lo < 1e-4:
                break
    _SIGMA_CACHE[key] = hi
    return hi


def _eps(A, sigma, q, steps, delta):
    a = A()
    a.history = [(sigma, q, steps)]
    return float(a.get_epsilon(delta=delta))


def epsilon_for(sigma, q, steps, delta):
    """(ε_PRV, ε_RDP) for `steps` Poisson-subsampled Gaussian steps."""
    if steps == 0:
        return 0.0, 0.0
    from opacus.accountants import PRVAccountant, RDPAccountant
    return _eps(PRVAccountant, sigma, q, steps, delta), _eps(RDPAccountant, sigma, q, steps, delta)


def epsilon_rdp(sigma, q, steps, delta):
    """RDP only: cheap enough to log every round (PRV costs ~0.5-2 s per call)."""
    from opacus.accountants import RDPAccountant
    return _eps(RDPAccountant, sigma, q, steps, delta) if steps else 0.0


class ClientAccountant:
    def __init__(self, sigma, q, delta):
        self.sigma, self.q, self.delta, self.steps = sigma, q, delta, 0

    def step(self, n=1):
        self.steps += n

    def epsilon(self):
        """-> (ε_PRV, ε_RDP) spent so far."""
        return epsilon_for(self.sigma, self.q, self.steps, self.delta)

    def epsilon_rdp(self):
        return epsilon_rdp(self.sigma, self.q, self.steps, self.delta)


def _unit_grad_chunks(model, loss_fn, F, T, Y, Mk, IM, W=None, chunk=128):
    """Yields per-unit (patient/image) gradients of the weighted unit-mean loss, `chunk` units at a time."""
    params = {k: v.detach() for k, v in model.named_parameters()}
    W = torch.ones_like(IM) if W is None else W

    def unit_loss(p, f, t, y, m, im, w):
        logits = functional_call(model, p, (f, t))
        return (loss_fn(logits, y, m) * w * im).sum() / im.sum().clamp(min=1)

    g = vmap(grad(unit_loss), in_dims=(None, 0, 0, 0, 0, 0, 0), randomness="different")
    for i in range(0, F.shape[0], chunk):
        yield g(params, F[i:i + chunk], T[i:i + chunk], Y[i:i + chunk], Mk[i:i + chunk], IM[i:i + chunk], W[i:i + chunk])


def unit_grads(model, loss_fn, F, T, Y, Mk, IM, W=None, chunk=128):
    """All per-unit gradients: dict name -> [B, *shape] (tests/diagnostics; training streams via dp_gradient)."""
    outs = list(_unit_grad_chunks(model, loss_fn, F, T, Y, Mk, IM, W, chunk))
    return {k: torch.cat([o[k] for o in outs]) for k in outs[0]}


def clip_and_sum(grads, C):
    """Clip each unit's total gradient to norm C and sum over units. Returns (sum dict, norms).
    No temporary the size of the per-unit gradients: norms by fused reductions, the clipped sum as one contraction over units.
    This halves peak memory so a larger vmap chunk fits; the step is bound by kernel launches, so fewer, larger chunks are faster."""
    norms = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.flatten(1), dim=1) for g in grads.values()], 1), dim=1)
    factor = (C / (norms + 1e-6)).clamp(max=1.0)
    return {k: torch.tensordot(factor, g, dims=1) for k, g in grads.items()}, norms


def dp_gradient(model, loss_fn, batch, C, sigma, B_expected, chunk=128):
    """Sets p.grad = (Σ_u clip(g_u) + N(0, σ²C²)) / B_expected. batch=None -> empty Poisson sample (noise only).
    Returns the fraction of units that were clipped as a 0-d device tensor (no host sync), or None for an empty sample."""
    summed = {k: torch.zeros_like(p) for k, p in model.named_parameters()}
    frac = None
    if batch is not None:                       # clip + sum chunk by chunk: memory scales with `chunk`, not batch size
        n_clip = 0
        for g in _unit_grad_chunks(model, loss_fn, *batch, chunk=chunk):
            s, norms = clip_and_sum(g, C)
            for k in summed:
                summed[k] += s[k]
            n_clip = n_clip + (norms > C).sum()
            del g, s
        frac = n_clip / max(batch[0].shape[0], 1)
    for k, p in model.named_parameters():
        g = summed[k]
        if sigma > 0:
            g = g + torch.randn_like(g) * sigma * C
        p.grad = g / B_expected
    return frac
