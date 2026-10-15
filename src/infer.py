"""Run an existing checkpoint on the locked test split without training."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
import common as C
from models import PGF, build


def load_checkpoint(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    # torch.compile wraps individual Mamba blocks with _orig_mod. The tensor
    # values are unchanged; normalize those names for portable eager inference.
    return {key.replace('._orig_mod.', '.').removeprefix('_orig_mod.'): value
            for key, value in state.items()}


def model_for_stem(stem):
    if stem.startswith('pgf'):
        tpa = '_tpa' in stem
        return PGF(use_csi='_nocsi' not in stem, use_gate='_nogate' not in stem,
                   use_image='_noimg' not in stem, use_fd='_fd' in stem,
                   use_deep='_deep' in stem, use_cl_b='_clb' in stem,
                   use_awa_b=('_awa' in stem or tpa),
                   awa_align=not ('_awana' in stem or tpa),
                   awa_extrap='_awane' not in stem,
                   awa_near=not ('_awanr' in stem or tpa),
                   tpa_ms='_tpams' in stem, tpa_sa='_tpasa' in stem), 'pgf'
    if stem.startswith('skybimamba'):
        from skybimamba import SkyBiMamba
        return SkyBiMamba(), 'sky'
    return build(stem.split('_')[0]), 'baseline'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family', default='tpa-rw', choices=['tpa', 'tpa-rw', 'tpa-ms', 'tpa-sa'])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--horizon', type=int, choices=[5, 15, 30], default=15)
    ap.add_argument('--checkpoint', type=Path, help='One explicit *_ckpt.pt path; overrides family')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    ap.add_argument('--limit', type=int, help='Smoke test only: first N test windows')
    ap.add_argument('--check-stored', action='store_true')
    ap.add_argument('--output', type=Path, default=Path(C.ROOT) / 'outputs/inference')
    args = ap.parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    names = {'tpa': 'pgf_nocsi_tpa', 'tpa-rw': 'pgf_nocsi_tpa_rw4',
             'tpa-ms': 'pgf_nocsi_tpa_tpams', 'tpa-sa': 'pgf_nocsi_tpa_tpasa'}
    suffix = '' if args.horizon == 15 else f'_h{args.horizon}'
    paths = [args.checkpoint] if args.checkpoint else [Path(C.RESULTS_DIR) /
             f'{names[args.family]}{suffix}_s{s}_ckpt.pt' for s in args.seeds]
    predictions = []
    horizons = set()
    args.output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        stem = path.name.removesuffix('_ckpt.pt')
        match = re.search(r'_h(5|30)(?:_|$)', stem)
        horizon = int(match.group(1)) if match else 15
        horizons.add(horizon)
        model, kind = model_for_stem(stem)
        model.load_state_dict(load_checkpoint(path), strict=True)
        model.to(device).eval()
        split = C.get_split('test', horizon)
        idx = np.arange(min(args.limit, len(split['tgt']))) if args.limit else None
        base = stem.split('_')[0]
        ds = C.ForecastDataset('test', idx, csi=kind != 'baseline',
                               no_image=base in ['mlp', 'gru', 'dlinear', 'patchtst', 'crossgnn'],
                               time_feats=kind == 'sky', horizon=horizon)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        parts = []
        with torch.inference_mode():
            for batch in loader:
                if kind == 'pgf':
                    image, pv, csi, _, pt_next, _ = batch
                    pred = model(image.to(device), pv.to(device), csi.to(device), pt_next.to(device))
                    if '_res' in stem:
                        pred = pred + pv[:, -1].to(device)
                elif kind == 'sky':
                    image, pv, _, _, _, _, time_feats = batch
                    pred = model(image.to(device), pv.to(device), time_feats.to(device))
                else:
                    image, pv, _ = batch
                    pred = model(image.to(device), pv.to(device))
                parts.append(pred.cpu().numpy())
        pred = np.concatenate(parts).astype(np.float64)
        if args.check_stored:
            expected = np.load(Path(C.RESULTS_DIR) / f'preds_{stem}.npy')[:len(pred)]
            max_error = float(np.max(abs(pred - expected)))
            # Check CPU/GPU and convolution algorithm floating-point differences.
            if not np.allclose(pred, expected, rtol=1e-4, atol=1e-3):
                raise AssertionError(f'{stem}: stored prediction mismatch, max={max_error}')
            print(f'PASS {stem}: max abs difference {max_error:.8f} kW', flush=True)
        np.save(args.output / f'preds_{stem}.npy', pred)
        predictions.append(pred)
    if len(horizons) != 1:
        raise ValueError('Cannot ensemble different horizons')
    ensemble = np.mean(predictions, axis=0)
    n = len(ensemble)
    y = split['pv'][split['tgt']][:n]
    report = C.evaluate(y, ensemble, split['anchor_time'][:n])
    report.update(horizon=horizon, checkpoints=[p.name for p in paths], smoke_test=bool(args.limit))
    np.save(args.output / 'ensemble.npy', ensemble)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
