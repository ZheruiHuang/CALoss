"""Load prepared OD tensors from /data/<dataset>/{train,valid,test}.pt."""
from pathlib import Path
import torch
from losses import training_flow_unit


def load_split(root, split):
    rows = torch.load(Path(root) / f"{split}.pt", map_location="cpu", weights_only=True)
    for s in rows:
        for key in ("features", "coordinates", "distance"):
            s[key] = s[key].float()
        s["target"] = s["target"].double()
        s["mask"] = s["mask"].bool()
        s["pack"] = {k: v.float() for k, v in s["pack"].items()}
        n = len(s["features"])
        assert s["target"].shape == s["mask"].shape == s["distance"].shape == (n, n)
        assert s["coordinates"].shape == (n, 2)
        assert torch.isfinite(s["target"]).all() and (s["target"] >= 0).all()
        assert s["target"][s["mask"]].sum() > 0
    if not rows:
        raise ValueError(f"Empty {split} split")
    return rows


def scales(train, regime):
    from models.cross import WeDAN
    return dict(reference_rate=training_flow_unit(train),
                distance_max=max(float(s["distance"].max()) for s in train),
                flow_log1p_max=max(float(torch.log1p(WeDAN.target_flow(
                    s["target"], s["mask"], regime)[s["mask"]]).max()) for s in train))


def prepare(rows, scale, cross):
    if cross:
        for s in rows:
            s["distance"] = 2 * s["distance"] / max(scale["distance_max"], 1e-8) - 1
    return rows
