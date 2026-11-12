"""Baseline losses, CALoss, and training-data calibration."""
import math
import numpy as np
import torch
import torch.nn.functional as F

def overlap_norm(error, p=1):
    """Lp norm of flow residuals."""
    return error.abs().sum() if p == 1 else torch.linalg.vector_norm(error, ord=p)

def decode_loss(logits, target, mask, regime, loss_name, alpha, reference_rate, huber_beta=1.0, pointwise_unit=1.0, overlap_p=1):
    """Decode flows and compute one instance's loss."""
    z = logits[mask].to(torch.float64)
    y = target.to(dtype=torch.float64, device=z.device)[mask]
    mass = y.sum()
    if regime == "known":
        log_prediction = torch.log_softmax(z, dim=0)
        normalized_prediction = log_prediction.exp()
        normalized_target = y / mass
        prediction = normalized_prediction * mass
        overlap = overlap_norm(normalized_prediction - normalized_target, overlap_p) / 2
        positive = normalized_target > 0
        deviance = (normalized_target[positive] *
                    (normalized_target[positive].log() - log_prediction[positive])).sum()
        # Unit = mean admissible flow, m/N. This avoids a probability-scale
        # Huber threshold that would put every residual in its quadratic branch.
        point_prediction = normalized_prediction * y.numel()
        point_target = normalized_target * y.numel()
    else:
        log_prediction = (z + math.log(reference_rate)).clamp(-20, 20)
        prediction = log_prediction.exp()
        overlap = overlap_norm(prediction - y, overlap_p) / (prediction.sum() + mass)
        positive = y > 0
        deviance = ((y[positive] * (y[positive].log() - log_prediction[positive])).sum()
                    - mass + prediction.sum()) / mass
        if pointwise_unit == 1.0:
            point_prediction, point_target = torch.log1p(prediction), torch.log1p(y)
        else:
            point_prediction, point_target = torch.log1p(prediction / pointwise_unit), torch.log1p(y / pointwise_unit)
    if loss_name == "caloss":
        value = (1 - alpha) * overlap + alpha * deviance
    elif loss_name == "l1":
        value = F.l1_loss(point_prediction, point_target)
    elif loss_name == "l2":
        value = F.mse_loss(point_prediction, point_target)
    elif loss_name == "huber":
        value = F.smooth_l1_loss(point_prediction, point_target, beta=huber_beta)
    else:
        raise ValueError(loss_name)
    return value, prediction, y

def training_prior(samples, regime, overlap_p=1):
    totals = [float(s["target"][s["mask"]].sum()) for s in samples]
    median_total = float(np.median(totals))
    overlaps, deviances = [], []
    for sample, mass in zip(samples, totals):
        y = sample["target"][sample["mask"]].double()
        prediction = torch.full_like(y, (mass if regime == "known" else median_total) / len(y))
        overlap = overlap_norm(prediction - y, overlap_p) / (prediction.sum() + mass)
        positive = y > 0
        deviance = ((y[positive] * (y[positive].log() - prediction[positive].log())).sum()
                    - mass + prediction.sum()) / mass
        overlaps.append(float(overlap))
        deviances.append(float(deviance))
    left, right = float(np.median(overlaps)), float(np.median(deviances))
    return {"alpha": left / (left + right) if left + right > 1e-12 else 0.5,
            "median_overlap": left, "median_deviance": right,
            "median_training_total": median_total}

def training_flow_unit(samples):
    rate = float(np.median([float(s["target"][s["mask"]].double().mean()) for s in samples]))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("Training mean-flow unit must be finite and positive")
    return rate

def batch_inputs(samples, device):
    return ([s["features"].to(device) for s in samples],
            [s["coordinates"].to(device) for s in samples],
            [{k: v.to(device) for k, v in s["pack"].items()} for s in samples])

def with_reference(output, sample):
    if "log_reference" not in sample:
        return output
    return output.double() + sample["log_reference"].to(device=output.device, dtype=torch.float64)
