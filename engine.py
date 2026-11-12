"""Backbone training and prediction."""
import torch
import torch.nn.functional as F
import losses as helpers
from models.cross import DeepGravity, WeDAN, dgm_objective
from models.prediction import PredODGModel
from models.generation import FMODGModel

def objective(model, sample, args, scale, rng):
    x, y, mask, d = [sample[k].to(args.device) for k in ["features", "target", "mask", "distance"]]
    if args.model == "dgm":
        if args.regime == "unknown":
            return dgm_objective(model(x, d), y, mask, args.loss, args.alpha, args.regime, args.reference_rate)[0]
        if args.loss == "caloss" or args.loss == "l2":
            return dgm_objective(model(x, d), y, mask, args.loss, args.alpha)[0]
        _, pred, _ = dgm_objective(model(x, d), y, mask)
        unit = y[mask].sum() / mask.sum()
        return (F.l1_loss if args.loss == "l1" else F.smooth_l1_loss)(pred[mask] / unit, y.double()[mask] / unit)
    if args.loss == "caloss":
        return model.objective(x, y, mask, d, scale['flow_log1p_max'], rng, weight=1., alpha=args.alpha, regime=args.regime)[0]
    clean = 2 * torch.log1p(model.target_flow(y, mask, args.regime).float()) / scale['flow_log1p_max'] - 1
    t = torch.randint(len(model.abar), (1,), generator=rng, device=y.device)
    noise = torch.randn(clean.shape, generator=rng, device=y.device)
    a = model.abar[t]
    state = (a.sqrt() * clean + (1-a).sqrt() * noise).masked_fill(~mask, 0.)
    estimate = model(x, state, d, t).float()
    native = (estimate[mask] - noise[mask]).square().mean()
    prediction = model.decode((state - (1-a).sqrt() * estimate) / a.sqrt(), scale['flow_log1p_max'], mask, y[mask].sum() if args.regime == 'known' else None)[mask]
    unit = y[mask].sum() / mask.sum() if args.regime == 'known' else args.reference_rate
    loss = {'l1': F.l1_loss, 'l2': F.mse_loss, 'huber': F.smooth_l1_loss}[args.loss]
    return native + loss(prediction / unit, y.double()[mask] / unit)


def build(args, features):
    if args.model == 'dgm':
        return DeepGravity(features=features)
    if args.model == 'wedan':
        return WeDAN(features=features)
    kw = dict(feat_dim=features, hid_dim=32, encode_mlp_layers=1,
              encode_gattn_layers=1, attn_heads=4, attn_layers=1,
              axial_attn_layers=1, dropout=0.)
    return PredODGModel(**kw) if args.task == 'prediction' else FMODGModel(time_dim=32, **kw)


def decoded(latent, s, args):
    logits = latent.double().clamp(1e-8, 20).expm1().log()
    return helpers.decode_loss(logits, s['target'], s['mask'].to(args.device),
        args.regime, args.loss, args.alpha, args.reference_rate, overlap_p=args.overlap_p)


def loss(model, s, args, scale):
    if args.model != 'geoflow':
        return objective(model, s, args, scale, None)
    inputs = helpers.batch_inputs([s], args.device)
    mask = s['mask'].to(args.device)
    if args.task == 'prediction':
        logits = helpers.with_reference(model(*inputs)[0], s)
        return helpers.decode_loss(logits, s['target'], mask, args.regime,
            args.loss, args.alpha, args.reference_rate, overlap_p=args.overlap_p)[0]
    y = s['target'].to(args.device)
    unit = y[mask].mean() if args.regime == 'known' else args.reference_rate
    clean = torch.log1p(y / unit).float().masked_fill(~mask, 0.)
    noise = torch.randn_like(clean).masked_fill(~mask, 0.)
    t = torch.rand(1, device=args.device)
    state = (1-t)*noise + t*clean
    velocity = model([state], *inputs, [t])[0]
    native = (velocity[mask] - (clean-noise)[mask]).square().mean()
    return native + decoded(state + (1-t)*velocity, s, args)[0]


@torch.no_grad()
def predict(model, s, args, scale):
    mask = s['mask'].to(args.device)
    y = s['target'].to(args.device)
    if args.model == 'dgm':
        z = model(s['features'].to(args.device), s['distance'].to(args.device))
        return dgm_objective(z, y, mask, 'caloss', args.alpha, args.regime,
                             args.reference_rate)[1][mask]
    if args.model == 'wedan':
        return model.sample(s['features'].to(args.device), s['distance'].to(args.device),
            mask, scale['flow_log1p_max'], None, 25,
            y[mask].sum() if args.regime == 'known' else None)[mask]
    inputs = helpers.batch_inputs([s], args.device)
    if args.task == 'prediction':
        z = helpers.with_reference(model(*inputs)[0], s)
        return helpers.decode_loss(z, y, mask, args.regime, args.loss,
            args.alpha, args.reference_rate, overlap_p=args.overlap_p)[1]
    state = torch.randn(mask.shape, device=args.device).masked_fill(~mask, 0.)
    for step in range(25):
        t = torch.tensor([step/25], device=args.device)
        state = (state + model([state], *inputs, [t])[0]/25).masked_fill(~mask, 0.)
    return decoded(state, s, args)[1]
