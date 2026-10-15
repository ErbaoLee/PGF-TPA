"""Download the exact upstream Parquet revision recorded by the experiments."""
import argparse
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = 'f51690de73d34cc409670ef6119fd69cc3aade8d'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split', choices=['train', 'test', 'all'], default='all')
    ap.add_argument('--output', type=Path, default=ROOT / 'data/raw')
    args = ap.parse_args()
    info = json.loads((ROOT / 'data/upstream_dataset_info.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for ref, spec in info['download_checksums'].items():
        name = ref.rsplit('/', 1)[-1]
        if args.split != 'all' and not name.startswith(args.split + '-'):
            continue
        path = args.output / name
        expected = spec['num_bytes']
        if path.exists() and path.stat().st_size == expected:
            print('Present:', name)
            continue
        url = f'https://huggingface.co/datasets/solarbench/SKIPPD/resolve/{REVISION}/data/{name}'
        tmp = path.with_suffix(path.suffix + '.partial')
        print('Downloading:', name, flush=True)
        with requests.get(url, stream=True, timeout=(30, 180)) as response:
            response.raise_for_status()
            with tmp.open('wb') as stream:
                for block in response.iter_content(1024 * 1024):
                    if block:
                        stream.write(block)
        if tmp.stat().st_size != expected:
            raise RuntimeError(f'{name}: size mismatch, expected {expected}')
        import pyarrow.parquet as pq
        pq.read_metadata(tmp)
        tmp.replace(path)
    print('Download complete. Run python src/prepare_data.py')


if __name__ == '__main__':
    main()
