"""Update compression for FL traffic.

encode() -> payload tensors; serialize() -> the bytes actually transmitted (byte counts are len(bytes), not assumed);
the server deserializes and aggregates the DEQUANTISED tensors. Everything here acts on the client's update after DP noise,
so it is post-processing and cannot change ε.

Modes: fp32 | fp16 | int8 (symmetric per-block, fp32 scales) | topk (top-k magnitudes, fp16 values + int32 indices,
with client-side error feedback whose residual is built from the noisy update only).
"""
import json, math, struct
import numpy as np
import torch

MODES = ("fp32", "fp16", "int8", "topk")


def encode(vec, mode, block=1024, topk=0.01):
    """vec: flat float tensor -> payload dict of tensors + plain ints."""
    vec = vec.detach().float()
    n = vec.numel()
    if mode == "fp32":
        return {"x": vec.cpu(), "n": n}
    if mode == "fp16":
        return {"x": vec.half().cpu(), "n": n}
    if mode == "int8":
        v = torch.nn.functional.pad(vec, (0, (-n) % block)).view(-1, block)
        scale = (v.abs().amax(1, keepdim=True) / 127).clamp(min=1e-12)
        q = torch.round(v / scale).clamp(-127, 127).to(torch.int8)
        return {"q": q.cpu(), "s": scale.flatten().cpu(), "n": n, "block": block}
    if mode == "topk":
        k = max(1, math.ceil(topk * n))
        idx = vec.abs().topk(k).indices.sort().values
        return {"i": idx.to(torch.int32).cpu(), "v": vec[idx].half().cpu(), "n": n}
    raise ValueError(mode)


def decode(payload, mode, device="cpu"):
    n = payload["n"]
    if mode in ("fp32", "fp16"):
        out = payload["x"].float()
    elif mode == "int8":
        out = (payload["q"].float().view(-1, payload["block"]) * payload["s"].view(-1, 1)).flatten()[:n]
    elif mode == "topk":
        out = torch.zeros(n)
        out[payload["i"].long()] = payload["v"].float()
    else:
        raise ValueError(mode)
    return out.to(device)


def serialize(payload):
    """Compact wire format: 4-byte header length | JSON header (names, dtypes, shapes, ints) | raw little-endian buffers."""
    head, bufs = {}, []
    for k, v in payload.items():
        if torch.is_tensor(v):
            a = v.contiguous().numpy()
            head[k] = {"dtype": a.dtype.str, "shape": list(a.shape)}
            bufs.append(a.tobytes())
        else:
            head[k] = {"int": int(v)}
    h = json.dumps(head, separators=(",", ":")).encode()
    return struct.pack("<I", len(h)) + h + b"".join(bufs)


def deserialize(blob):
    (hl,) = struct.unpack("<I", blob[:4])
    head = json.loads(blob[4:4 + hl])
    off, out = 4 + hl, {}
    for k, d in head.items():
        if "int" in d:
            out[k] = d["int"]
            continue
        dt = np.dtype(d["dtype"])
        cnt = int(np.prod(d["shape"])) if d["shape"] else 1
        a = np.frombuffer(blob, dtype=dt, count=cnt, offset=off).reshape(d["shape"])
        out[k] = torch.from_numpy(a.copy())
        off += cnt * dt.itemsize
    return out


def transmit(vec, mode, cfg_comm, device):
    """Encode -> bytes -> decode, exactly what the receiver reconstructs. Returns (reconstructed vec, nbytes)."""
    blob = serialize(encode(vec, mode, cfg_comm.get("block", 1024), cfg_comm.get("topk", 0.01)))
    return decode(deserialize(blob), mode, device), len(blob)


class ErrorFeedback:
    """Per-client residual for biased compressors (top-k). residual = (noisy update + old residual) - what was sent."""

    def __init__(self):
        self.res = {}

    def compress(self, k, delta, mode, cfg_comm):
        v = delta + self.res.get(k, torch.zeros_like(delta)) if mode == "topk" else delta
        sent, nb = transmit(v, mode, cfg_comm, delta.device)
        if mode == "topk":
            self.res[k] = v - sent
        return sent, nb
