"""Local training for one FL client: non-DP AdamW (AMP) or patient/image-level DP-SGD in fp32 (docs/05, 06)."""
import torch
from torch.nn.utils import parameters_to_vector
from ..data.store import gather_units
from .losses import weighted_mean
from .privacy import dp_gradient


def local_train(model, client, cfg, steps, loss_fn, gen, dp=None):
    """Runs `steps` local steps in place. dp: dict(sigma, q, unit, M, C, B, lr, acct) or None.
    Returns (flat parameter delta, mean clipped fraction or None)."""
    store = client.store
    w0 = parameters_to_vector(model.parameters()).detach().clone()
    model.train()
    fl = cfg["fl"]
    if dp:
        opt = torch.optim.SGD(model.parameters(), lr=dp["lr"], momentum=0.0)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=fl["lr"], weight_decay=fl["wd"])
    amp = fl.get("amp", True) and store.device.type == "cuda"
    clipped = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        if dp:
            s = client.sample_units(gen, dp["q"], dp["unit"], dp["M"])
            batch = gather_units(store, *s) if s is not None else None
            clipped.append(dp_gradient(model, loss_fn, batch, dp["C"], dp["sigma"], dp["B"], cfg["dp"]["vmap_chunk"]))
            dp["acct"].step()
        else:
            x, t, y, m, w = store.batch(client.sample_batch(gen, fl["batch"]))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                z = model(x, t)
            weighted_mean(loss_fn(z.float(), y, m), w).backward()
        opt.step()
    frac = [c for c in clipped if c is not None]                          # device tensors: one host sync per local round
    return parameters_to_vector(model.parameters()).detach() - w0, (float(torch.stack(frac).float().mean()) if frac else None)
