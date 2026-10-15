"""Run documented experiment recipes without overwriting frozen release results."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--family', default='tpa-rw', choices=['tpa', 'tpa-rw', 'tpa-ms', 'tpa-sa',
                    'cnn', 'cnn-rw', 'clb', 'clb-rw', 'mlp', 'gru', 'sunset', 'convlstm',
                    'reslstm', 'transformer', 'patchtst', 'dlinear', 'crossgnn', 'skybimamba'])
    ap.add_argument('--horizons', type=int, nargs='+', choices=[5, 15, 30], default=[15])
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--epochs', type=int)
    ap.add_argument('--batch-size', type=int)
    ap.add_argument('--output', type=Path, default=ROOT / 'outputs/retrained')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    env = os.environ.copy()
    env['PGF_RESULTS_DIR'] = str(args.output.resolve())
    for horizon in args.horizons:
        for seed in args.seeds:
            family = args.family
            if family in ['tpa', 'tpa-rw', 'tpa-ms', 'tpa-sa', 'cnn', 'cnn-rw', 'clb', 'clb-rw']:
                entry, flags, epochs, batch = 'run_pgf.py', ['--no-csi'], 40, 128
                if family.startswith('tpa'):
                    flags += ['--tpa']
                if family.startswith('clb'):
                    flags += ['--cl-b']
                if family.endswith('-rw'):
                    flags += ['--ramp-w', '4']
                if family in ['tpa-ms', 'tpa-sa']:
                    flags += ['--' + family]
            elif family == 'skybimamba':
                entry, flags, epochs, batch = 'run_skybimamba.py', ['--compile', '0', '--amp', 'off'], 40, 256
            else:
                entry, flags, epochs, batch = 'run_baselines.py', ['--model', family], 60, 256
            cmd = [sys.executable, str(ROOT / 'src' / entry), *flags,
                   '--horizon', str(horizon), '--seed', str(seed),
                   '--epochs', str(args.epochs or epochs), '--batch-size', str(args.batch_size or batch)]
            print(subprocess.list2cmdline(cmd), flush=True)
            if not args.dry_run:
                args.output.mkdir(parents=True, exist_ok=True)
                subprocess.run(cmd, env=env, check=True, cwd=ROOT)


if __name__ == '__main__':
    main()
