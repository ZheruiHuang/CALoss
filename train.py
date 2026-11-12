"""Train, select on validation data, and evaluate an OD model."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import random
import torch
import data
import engine
from losses import training_prior
from metrics import measure

BASELINES = ['l1', 'l2', 'huber', 'deviance', 'cpc']


@torch.no_grad()
def evaluate(model, samples, args, scale, draws=1):
    model.eval()
    rows = [measure(engine.predict(model, s, args, scale), s['target'][s['mask']])
            for s in samples for _ in range(draws)]
    return {k: sum(r[k] for r in rows)/len(rows) for k in ('CPC', 'NRMSE', 'JSD_allocation_bits')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['geoflow', 'dgm', 'wedan'], default='geoflow')
    parser.add_argument('--task', choices=['prediction', 'generation'], default='prediction')
    parser.add_argument('--regime', choices=['known', 'unknown'], default='unknown')
    parser.add_argument('--method', choices=BASELINES+['cal_def', 'cal_cal', 'best_of_pool'], default='cal_def')
    parser.add_argument('--metric', choices=['CPC', 'NRMSE', 'JSD_allocation_bits'], default='CPC')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--p', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    if args.model != 'geoflow':
        args.task = 'prediction' if args.model == 'dgm' else 'generation'
        if args.p != 1: parser.error('p study uses GeoFlow')
    args.output.mkdir(parents=True, exist_ok=False)
    train, valid = data.load_split(args.data, 'train'), data.load_split(args.data, 'valid')
    scale = data.scales(train, args.regime)
    prior = training_prior(train, args.regime, args.p)['alpha']
    args.reference_rate = scale['reference_rate']
    train, valid = [data.prepare(rows, scale, args.model != 'geoflow') for rows in (train, valid)]
    if args.method == 'cal_cal':
        candidates = [('cal_cal', a) for a in dict.fromkeys(min(1., max(0., prior+k*.15)) for k in range(-2, 3))]
    else:
        candidates = [(m, {'cpc': 0., 'deviance': 1.}.get(m, .5)) for m in
                      (BASELINES if args.method == 'best_of_pool' else [args.method])]
    initial = engine.build(args, train[0]['features'].shape[1]).to(args.device)
    sign = 1 if args.metric == 'CPC' else -1
    selected, history = None, []
    for index, (method, alpha) in enumerate(candidates):
        args.alpha, args.loss = alpha, method if method in ['l1', 'l2', 'huber'] else 'caloss'
        args.overlap_p = args.p if method in ('cal_def', 'cal_cal') else 1
        model = deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        directory = args.output / str(index)
        directory.mkdir()
        best = -float('inf')
        for epoch in range(1, args.epochs+1):
            model.train()
            random.shuffle(train)
            for start in range(0, len(train), args.batch_size):
                optimizer.zero_grad(set_to_none=True)
                value = torch.stack([engine.loss(model, s, args, scale) for s in train[start:start+args.batch_size]]).mean()
                if not torch.isfinite(value): raise FloatingPointError('Nonfinite training loss')
                value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1. if args.model == 'wedan' else 5., error_if_nonfinite=True)
                optimizer.step()
            scores = evaluate(model, valid, args, scale)
            row = dict(candidate=index, method=method, alpha=alpha, epoch=epoch, **scores)
            history.append(row)
            print(json.dumps(row), flush=True)
            quality = sign*scores[args.metric]
            if quality > best:
                best = quality
                torch.save(model.state_dict(), directory/'best.pt')
                best_row = row
            if epoch % 10 == 0:
                torch.save(model.state_dict(), directory/f'epoch_{epoch}.pt')
        if selected is None or best > selected[0]:
            selected = best, index, method, alpha, best_row
    _, index, method, args.alpha, row = selected
    args.loss = method if method in ['l1', 'l2', 'huber'] else 'caloss'
    args.overlap_p = args.p if method in ('cal_def', 'cal_cal') else 1
    initial.load_state_dict(torch.load(args.output/str(index)/'best.pt', map_location=args.device, weights_only=True))
    test = data.prepare(data.load_split(args.data, 'test'), scale, args.model != 'geoflow')
    result = dict(selection=row, test=evaluate(initial, test, args, scale, 3 if args.task == 'generation' else 1))
    (args.output/'history.json').write_text(json.dumps(history, indent=2))
    (args.output/'result.json').write_text(json.dumps(result, indent=2))
    (args.output/'config.json').write_text(json.dumps(dict(vars(args), data=str(args.data), output=str(args.output), scale=scale), indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
