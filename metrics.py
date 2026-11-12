"""CPC, NRMSE, and allocation JSD."""
import math
import torch

def measure(prediction, target):
    p, y = prediction.detach().cpu().double(), target.detach().cpu().double()
    if p.ndim != 1 or p.shape != y.shape or not p.numel():
        raise ValueError("Metrics require matching nonempty admissible flow vectors")
    if not torch.isfinite(p).all() or not torch.isfinite(y).all() or (p < 0).any() or (y < 0).any():
        raise ValueError("Metrics require finite nonnegative flows")
    predicted_mass, mass = float(p.sum()), float(y.sum())
    denominator = predicted_mass+mass
    cpc = float(2*torch.minimum(p, y).sum()/denominator) if denominator else 1.
    if not -1e-12 <= cpc <= 1+1e-12:
        raise ValueError("CPC left its mathematical range")
    rmse = float((p-y).square().mean().sqrt())
    nrmse = rmse/max(float(y.std(unbiased=False)), 1e-8, 1e-8*float(y.mean())) if mass else None
    jsd = None
    if mass and predicted_mass:
        q, qhat = y/mass, p/predicted_mass
        midpoint = (q+qhat)/2
        jsd = 0.
        for distribution in [q, qhat]:
            positive = distribution > 0
            jsd += float((distribution[positive]*(distribution[positive].log()-midpoint[positive].log())).sum()/(2*math.log(2)))
        if not -1e-12 <= jsd <= 1+1e-12:
            raise ValueError("Allocation JSD left its mathematical range")
        jsd = min(1., max(0., jsd))
    return {"CPC": min(1., max(0., cpc)), "RMSE": rmse, "NRMSE": nrmse,
            "JSD_allocation_bits": jsd,
            "relative_mass_error": abs(predicted_mass-mass)/mass if mass else None}
