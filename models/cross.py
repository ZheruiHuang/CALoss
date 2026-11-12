"""DGM and WeDAN models."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from models.wedan import DenoisingNet


class Recompute(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.inner = module

    def forward(self, *args):
        return checkpoint(self.inner, *args, use_reentrant=False) if self.training and torch.is_grad_enabled() else self.inner(*args)


class ChunkedCrossAttention(nn.Module):
    """Cross-attention computed in query chunks."""
    def __init__(self, module, rows=32):
        super().__init__()
        self.inner, self.rows = module, rows

    def forward(self, x, condition):
        shape = x.shape
        x, condition = x.reshape(-1, shape[-1]), condition.reshape(-1, shape[-1])
        module = self.inner
        q = module.q(x).reshape(len(x), module.n_head, module.df)
        k = module.k(condition).reshape(len(condition), module.n_head, module.df)
        v = module.v(condition).reshape(len(condition), module.n_head, module.df)
        def attend(qq, kk, vv):
            scores = qq[:, None] * kk[None] / math.sqrt(module.df)
            return (scores.softmax(1) * vv[None]).sum(1).flatten(1)
        pieces = []
        for start in range(0, len(x), self.rows):
            args = (q[start:start + self.rows], k, v)
            pieces.append(checkpoint(attend, *args, use_reentrant=False)
                          if self.training and torch.is_grad_enabled() else attend(*args))
        return (torch.cat(pieces) + x).reshape(shape)


class DeepGravity(nn.Module):
    """Pairwise MLP with five 256-unit and ten 128-unit hidden layers."""
    def __init__(self, features=131, dropout=.35, chunk_rows=16):
        super().__init__()
        dimensions = [2 * features + 1] + [256] * 5 + [128] * 10
        modules = []
        for a, b in zip(dimensions[:-1], dimensions[1:]):
            modules.extend([nn.Linear(a, b), nn.LeakyReLU(), nn.Dropout(dropout)])
        modules.append(nn.Linear(128, 1))
        self.network = nn.Sequential(*modules)
        self.chunk_rows = chunk_rows

    def forward(self, features, distance):
        n, pieces = len(features), []
        for start in range(0, n, self.chunk_rows):
            stop = min(start + self.chunk_rows, n)
            def chunk(x, d, start=start, stop=stop):
                origin = x[start:stop, None].expand(-1, n, -1)
                destination = x[None].expand(stop - start, -1, -1)
                inputs = torch.cat([origin, destination, d[start:stop, :, None]], dim=-1)
                return self.network(inputs).squeeze(-1)
            pieces.append(checkpoint(chunk, features, distance, use_reentrant=False)
                          if self.training and torch.is_grad_enabled() else chunk(features, distance))
        return torch.cat(pieces)


def dgm_objective(logits, target, mask, objective="native", alpha=.5, regime="known", reference_rate=1.):
    if regime == "unknown":
        from losses import decode_loss
        loss, values, _ = decode_loss(logits, target, mask, "unknown",
            "caloss" if objective == "native" else objective,
            1. if objective == "native" else alpha, reference_rate)
        prediction = torch.zeros_like(target, dtype=torch.float64).masked_scatter(mask, values)
        return loss, prediction, {}
    assert regime == "known"
    z, y = logits.double(), target.double().masked_fill(~mask, 0.)
    row_mass, mass = y.sum(1), y.sum()
    log_prob = z.masked_fill(~mask, -1e30).log_softmax(1)
    pred = log_prob.exp() * row_mass[:, None]
    pred = pred.masked_fill(~mask, 0.)
    active = row_mass > 0
    native = (-(y[active] / row_mass[active, None]) * log_prob[active]).sum(1).mean()
    overlap = (pred - y).abs().sum() / (2 * mass)
    pos = y > 0
    log_pred = log_prob + row_mass.clamp_min(1e-300).log()[:, None]
    deviance = (y[pos] * (y[pos].log() - log_pred[pos])).sum() / mass
    if objective == "native":
        loss = native
    elif objective == "caloss":
        loss = (1 - alpha) * overlap + alpha * deviance
    elif objective == "l2":
        unit = mass / mask.sum()
        loss = ((pred[mask] - y[mask]) / unit).square().mean()
    else:
        raise ValueError(objective)
    return loss, pred, {"native": native, "overlap": overlap, "deviance": deviance}


class WeDAN(nn.Module):
    def __init__(self, features=131, steps=250, recompute=True):
        super().__init__()
        self.config = dict(n_indim=features, n_outdim=features, e_indim=2, e_outdim=1,
                           hiddim=32, num_head=4, num_head_cross=1, num_layer=4,
                           dropout=0., norm_type="layer", LaPE_dim=0, T=steps)
        self.model = DenoisingNet(self.config)
        if recompute:
            transformer = self.model.graph_transformer
            transformer.gtlayers = nn.ModuleList([Recompute(m) for m in transformer.gtlayers])
            transformer.nftlayers = nn.ModuleList([Recompute(m) for m in transformer.nftlayers])
            transformer.crossAttns = nn.ModuleList([ChunkedCrossAttention(m) for m in transformer.crossAttns])
        def abar(t):
            return math.cos((t + .008) / 1.008 * math.pi / 2) ** 2
        betas = [min(1 - abar((i + 1) / steps) / abar(i / steps), .999) for i in range(steps)]
        self.register_buffer("abar", (1 - torch.tensor(betas, dtype=torch.float32)).cumprod(0))

    def forward(self, features, state, distance, time):
        # All attributes observed; no OD entries supplied at inference.
        node_input = torch.stack([features, torch.zeros_like(features)], dim=-1)
        edge_input = torch.stack([torch.zeros_like(state), state], dim=-1)
        masks = (torch.ones_like(features), torch.zeros_like(state))
        return self.model((node_input, edge_input), masks, distance, time,
                          torch.ones_like(state))[1]

    @staticmethod
    def target_flow(target, mask, regime):
        if regime == "unknown":
            return target
        assert regime == "known"
        return target / (target[mask].sum() / mask.sum())

    def objective(self, features, target, mask, distance, log_max, rng, weight=0., alpha=.5, regime="unknown"):
        clean = 2 * torch.log1p(self.target_flow(target, mask, regime).float()) / log_max - 1
        t = torch.randint(len(self.abar), (1,), generator=rng, device=target.device)
        noise = torch.randn(clean.shape, generator=rng, device=target.device)
        a = self.abar[t]
        state = a.sqrt() * clean + (1 - a).sqrt() * noise
        state = state.masked_fill(~mask, 0.)
        noise_hat = self(features, state, distance, t).float()
        native = (noise_hat[mask] - noise[mask]).square().mean()
        auxiliary = native.new_zeros(())
        if weight:
            clean_hat = (state - (1 - a).sqrt() * noise_hat) / a.sqrt()
            pred = self.decode(clean_hat, log_max, mask, target[mask].sum() if regime == "known" else None).double()[mask]
            y = target.double()[mask]
            overlap = (pred - y).abs().sum() / (pred.sum() + y.sum())
            pos = y > 0
            dev = ((y[pos] * (y[pos].log() - pred[pos].clamp_min(1e-12).log())).sum()
                   - y.sum() + pred.sum()) / y.sum()
            auxiliary = (1 - alpha) * overlap + alpha * dev
        return native + weight * auxiliary, {"native": native, "auxiliary": auxiliary}

    @staticmethod
    def decode(clean, log_max, mask=None, mass=None):
        prediction = torch.expm1((clean.clamp(-1, 1).double() + 1) * log_max / 2)
        if mass is not None:
            # Known global volume enters only this decoder and the training target.
            prediction = prediction.clamp_min(1e-12).masked_fill(~mask, 0.)
            prediction = prediction / prediction.sum() * mass
        return prediction

    @torch.no_grad()
    def sample(self, features, distance, mask, log_max, rng, sampling_steps, mass=None):
        if not 2 <= sampling_steps <= len(self.abar):
            raise ValueError("Invalid DDIM sampling steps")
        # Includes the final diffusion index and an explicit clean endpoint.
        indices = torch.linspace(len(self.abar) - 1, 0, sampling_steps).round().long().tolist()
        assert len(set(indices)) == len(indices)
        state = torch.randn(mask.shape, generator=rng, device=features.device).masked_fill(~mask, 0.)
        for i, t in enumerate(indices):
            noise = self(features, state, distance, torch.tensor([t], device=features.device)).float()
            a = self.abar[t]
            clean = ((state - (1 - a).sqrt() * noise) / a.sqrt()).clamp(-1, 1)
            if i + 1 == len(indices):
                state = clean
            else:
                previous = self.abar[indices[i + 1]]
                # Recompute epsilon after clipping so the update remains consistent.
                eps = (state - a.sqrt() * clean) / (1 - a).sqrt()
                state = previous.sqrt() * clean + (1 - previous).sqrt() * eps
            state = state.masked_fill(~mask, 0.)
        return self.decode(state, log_max, mask, mass).masked_fill(~mask, 0.)
