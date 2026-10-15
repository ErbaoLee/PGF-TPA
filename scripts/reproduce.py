"""Recompute public result tables from frozen predictions; NumPy only."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUNNY = {'2017-09-15', '2017-10-06', '2017-10-22', '2018-02-16', '2018-06-12',
         '2018-06-23', '2019-01-25', '2019-06-23', '2019-07-14', '2019-10-14'}
FAMILIES = {
    'TPA-E3': 'pgf_nocsi_tpa', 'TPA-RW-E3': 'pgf_nocsi_tpa_rw4',
    'Sky-BiMamba-E3': 'skybimamba', 'SUNSET-E3': 'sunset', 'ResLSTM-E3': 'reslstm',
    'ConvLSTM-E3': 'convlstm', 'Transformer-E3': 'transformer',
    'PatchTST-E3': 'patchtst', 'DLinear-E3': 'dlinear', 'CrossGNN-E3': 'crossgnn'}


def test_data(horizon, root=ROOT):
    meta = np.load(root / 'data/proc/test_meta.npz', allow_pickle=False)
    suffix = '' if horizon == 15 else f'_h{horizon}'
    samples = np.load(root / f'data/proc/test_samples{suffix}.npz', allow_pickle=False)
    times = samples['anchor_time']
    minutes = meta['minutes']
    assert np.all(np.diff(minutes[samples['win']], axis=1) == np.timedelta64(60, 's'))
    assert np.all(minutes[samples['win'][:, -1]] == times)
    assert np.all(minutes[samples['tgt']] == times + np.timedelta64(horizon, 'm'))
    y = meta['pv'][samples['tgt']].astype(np.float64)
    p0 = meta['pv'][samples['win'][:, -1]].astype(np.float64)
    sun = np.isin(times.astype('datetime64[D]').astype(str), list(SUNNY))
    masks = {'overall': np.ones(len(y), bool), 'sunny': sun, 'cloudy': ~sun,
             'ramp3kW': abs(y - p0) > 3, 'nonramp3kW': abs(y - p0) <= 3}
    return y, p0, masks


def load_prediction(stem, n, root=ROOT):
    path = root / 'results' / f'preds_{stem}.npy'
    pred = np.load(path, allow_pickle=False).astype(np.float64)
    if pred.shape != (n,) or not np.isfinite(pred).all():
        raise ValueError(f'Invalid prediction: {path.name}')
    return pred


def stats(y, pred, p0, mask):
    yy, err = y[mask], pred[mask] - y[mask]
    mse = np.mean(err ** 2)
    raw = np.sqrt(np.mean((p0[mask] - yy) ** 2))
    return dict(n=int(mask.sum()), rmse=float(np.sqrt(mse)), mae=float(abs(err).mean()),
                r2=float(1 - np.sum(err ** 2) / np.sum((yy - yy.mean()) ** 2)),
                fs_raw=float(1 - np.sqrt(mse) / raw))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/reproduction')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    expected = json.loads((ROOT / 'checkpoints/reference_metrics.json').read_text())
    for horizon in (5, 15, 30):
        y, raw, masks = test_data(horizon)
        suffix = '' if horizon == 15 else f'_h{horizon}'
        assert len(y) == {5: 12105, 15: 11180, 30: 9814}[horizon]
        predictions = {'RAW': raw}
        for family, base in FAMILIES.items():
            members = [load_prediction(f'{base}{suffix}_s{s}', len(y)) for s in range(3)]
            predictions[family] = np.mean(members, axis=0)
        if horizon == 15:
            folds = [load_prediction(f'pgf_nocsi_tpa_s{s}_f{f}', len(y))
                     for s in range(3) for f in range(5)]
            base = [load_prediction(f'pgf_nocsi_tpa_s{s}', len(y)) for s in range(3)]
            predictions['TPA-E18'] = np.mean(base + folds, axis=0)
            for label, base in [('TPA-MS-E3', 'pgf_nocsi_tpa_tpams'),
                                ('TPA-SA-E3', 'pgf_nocsi_tpa_tpasa')]:
                predictions[label] = np.mean([load_prediction(f'{base}_s{s}', len(y))
                                              for s in range(3)], axis=0)
        for label, pred in predictions.items():
            for subset, mask in masks.items():
                row = dict(model=label, horizon=horizon, subset=subset,
                           **stats(y, pred, raw, mask))
                key = f'{label}/{horizon}/{subset}'
                if key in expected:
                    for metric in ['rmse', 'mae', 'r2']:
                        if not np.isclose(row[metric], expected[key][metric], rtol=0, atol=1e-10):
                            raise AssertionError(f'Reference mismatch: {key}/{metric}')
                rows.append(row)
        print(f'PASS horizon={horizon}: {len(y)} samples, {len(predictions)} configurations')
    (args.output / 'metrics.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (args.output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Recomputed results', '', '| Model | Horizon (min) | RMSE (kW) | MAE (kW) | R² |',
             '|---|---:|---:|---:|---:|']
    for row in rows:
        if row['subset'] == 'overall':
            lines.append(f"| {row['model']} | {row['horizon']} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['r2']:.4f} |")
    (args.output / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('PASS: frozen reference metrics match. Outputs:', args.output)


if __name__ == '__main__':
    main()
