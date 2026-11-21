"""Grad-CAM through the chained model: frozen DenseNet encoder(s) -> head (docs/07 s7.4).
With several encoders, each encoder's CAM is computed from the same logit and the normalised maps are averaged."""
import torch
import torch.nn.functional as F


def gradcam(nets, head, x, tab, label_idx):
    """nets: list of encoders in head order; x [1,1,224,224]; tab [1,9]. Returns (cam [224,224] in 0..1, prob)."""
    head.eval()
    with torch.enable_grad():
        maps = [net.eval().features(x) for net in nets]            # [1,1024,7,7] each (post norm5)
        for A in maps:
            A.retain_grad()
        f = torch.stack([F.adaptive_avg_pool2d(F.relu(A), 1).flatten(1) for A in maps], dim=1)   # [1,E,1024]
        logit = head(f, tab)[0, label_idx]
        head.zero_grad()
        logit.backward()
    cams = []
    for A in maps:
        cam = F.relu((A.grad.mean((2, 3), keepdim=True) * A).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cams.append((cam - cam.min()) / (cam.max() - cam.min() + 1e-8))
    return torch.stack(cams).mean(0).detach().cpu().numpy(), float(torch.sigmoid(logit))
