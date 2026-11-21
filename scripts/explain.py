"""Grad-CAM grid + metadata ablation for one completed run, on VAL images (TEST is never re-read here).

  python scripts/explain.py --run E4_eps3__ab12cd34__s0 --label "Pleural Effusion" [--smoke]
"""
import sys
from pathlib import Path
import numpy as np
import torch
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dpmmfl.cli import parser, config
from dpmmfl.constants import LABELS
from dpmmfl.data.store import load_split
from dpmmfl.eval.gradcam import gradcam
from dpmmfl.eval.metrics import safe_auroc, mean_comp5, per_label
from dpmmfl.features.extract import ImgDS, build_encoder
from dpmmfl.models.head import build_head
from dpmmfl.train.infer import predict_logits
from dpmmfl.utils import get_device, save_json


def main():
    ap = parser(__doc__)
    ap.add_argument("--run", required=True, help="run_id under <results>/runs")
    ap.add_argument("--label", default="Pleural Effusion")
    a = ap.parse_args()
    run = Path(config(a)["results_dir"]) / "runs" / a.run
    cfg = yaml.safe_load((run / "config.yaml").read_text())
    dev = get_device(cfg)
    head = build_head(cfg).to(dev)
    head.load_state_dict(torch.load(run / "head.pt", map_location=dev))
    store = load_split(cfg, "train", dev)
    rows = np.flatnonzero((store.meta["split"] == "val").to_numpy())
    j = LABELS.index(a.label)
    z = predict_logits(head, store, rows)
    y, m = store.numpy_targets(rows)

    res = {"split": "VAL", "mean5_real_tab": mean_comp5(per_label(safe_auroc, y, z, m, 20))}
    if cfg["model"]["head"] == "film":
        real = store.tab.clone()
        store.tab.zero_()
        res["mean5_zero_tab"] = mean_comp5(per_label(safe_auroc, y, predict_logits(head, store, rows), m, 20))
        store.tab.copy_(real)
        with torch.no_grad():
            g, b = head.tab(store.tab[torch.as_tensor(rows, device=dev)]).chunk(2, -1)
        res["film_mean_abs_gamma"], res["film_mean_abs_beta"] = float(g.abs().mean()), float(b.abs().mean())
    save_json(res, run / "modality.json")
    print("modality:", res)

    order = np.argsort(-z[:, j])
    pos = [i for i in order if y[i, j] == 1 and m[i, j] > 0][:7]
    fp = [i for i in order if y[i, j] == 0 and m[i, j] > 0][:1]
    nets = [build_encoder(e, dev) for e in cfg["features"]["encoders"]]
    ds = ImgDS(store.meta["abs_path"].to_numpy()[rows])
    tab = store.tab[torch.as_tensor(rows, device=dev)]
    fig, axs = plt.subplots(2, 4, figsize=(14, 7.5))
    for ax, i in zip(axs.ravel(), pos + fp):
        x = ds[i][None].to(dev)
        cam, prob = gradcam(nets, head, x, tab[i:i + 1], j)
        ax.imshow(x[0, 0].cpu(), cmap="gray"); ax.imshow(cam, cmap="jet", alpha=0.4)
        ax.set_title(f"{'FALSE POS' if i in fp else 'true pos'}  p={prob:.2f}", fontsize=10); ax.axis("off")
    fig.suptitle(f"Grad-CAM (VAL): {a.label} - {a.run}")
    fig.tight_layout()
    out = run / f"gradcam_{a.label.replace(' ', '_')}.png"
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
