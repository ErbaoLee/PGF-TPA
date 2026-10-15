"""Check SHA-256 assets and, optionally, reconstructed data arrays and frames."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-dir', type=Path, default=ROOT / 'data/proc')
    ap.add_argument('--arrays', action='store_true', help='Check reconstructed arrays, independent of ZIP headers')
    ap.add_argument('--images', action='store_true', help='Check full reconstructed uint8 NPY files')
    ap.add_argument('--skip-assets', action='store_true', help='Only validate reconstructed data')
    args = ap.parse_args()
    if not args.skip_assets:
        assets = json.loads((ROOT / 'checkpoints/assets.json').read_text())
        for rel, spec in assets.items():
            path = ROOT / rel
            if not path.exists() or path.stat().st_size != spec['bytes'] or sha(path) != spec['sha256']:
                raise AssertionError(f'Asset mismatch: {rel}. Run git lfs pull for LFS pointer files.')
        print(f'PASS: {len(assets)} release assets match SHA-256')
    if args.arrays or args.images:
        import numpy as np
        manifest = json.loads((ROOT / 'data/manifest.json').read_text())
        if args.arrays:
            for rel, spec in manifest['array_sha256'].items():
                file, key = rel.split('/')
                arr = np.load(args.data_dir / file, allow_pickle=False)[key]
                assert list(arr.shape) == spec['shape'] and str(arr.dtype) == spec['dtype'], rel
                exact = hashlib.sha256(arr.tobytes()).hexdigest() == spec['sha256']
                if not exact and 'reference_key' in spec:
                    reference_path = ROOT / 'data/clearsky_reference.npz'
                    assert sha(reference_path) == manifest['clearsky_reference_sha256']
                    reference = np.load(reference_path)[spec['reference_key']]
                    assert np.allclose(arr, reference, rtol=0, atol=spec['absolute_tolerance']), rel
                else:
                    assert exact, rel
            print('PASS: all prepared arrays match the experiment data')
        if args.images:
            for split, spec in manifest['splits'].items():
                path = args.data_dir / f'{split}_images.dat.npy'
                arr = np.load(path, mmap_mode='r', allow_pickle=False)
                assert list(arr.shape) == spec['image_shape'], split
                assert sha(path) == spec['image_npy_sha256'], split
            print('PASS: complete image files match SHA-256')


if __name__ == '__main__':
    main()
