"""Reconstruct the locked minute frames and 5/15/30 minute forecast windows.

Accepts pinned upstream Parquet files or the original Hugging Face
save_to_disk Arrow directory. The datasets package is not required.
"""
import argparse
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def raw_files(source, split):
    state = source / split / 'state.json'
    if state.exists():
        entries = json.loads(state.read_text())['_data_files']
        return [source / split / row['filename'] for row in entries]
    paths = sorted(source.glob(f'{split}-*.parquet'))
    if not paths:
        raise FileNotFoundError(f'No {split} data in {source}; run scripts/download_data.py first')
    return paths


def batches(path, metadata_only=False):
    if path.suffix == '.arrow':
        with pa.memory_map(str(path), 'r') as mm:
            for batch in pa.ipc.open_stream(mm):
                yield batch.select(['time', 'pv']) if metadata_only else batch
    else:
        yield from pq.ParquetFile(path).iter_batches(
            batch_size=1024, columns=['time', 'pv'] if metadata_only else None)


def metadata(paths):
    tables = [pa.Table.from_batches(list(batches(path, True))) for path in paths]
    table = pa.concat_tables(tables)
    t = pd.Series(pd.to_datetime(table['time'].to_pandas(), utc=True)).dt.tz_convert('US/Pacific')
    pv = np.asarray(table['pv'], dtype=np.float32)
    order = np.argsort(t.to_numpy())
    sec = t.iloc[order].dt.tz_localize(None).to_numpy().astype('datetime64[s]').astype(np.int64)
    first = {}
    for k, second in enumerate(sec):
        first.setdefault(int(second), k)
    uni = np.array(sorted(first), dtype=np.int64)
    frame_rows = np.array([first[int(s)] for s in uni], dtype=np.int64)
    return uni, frame_rows, pv[order][frame_rows], order[frame_rows], len(pv)


def build_windows(uni, horizon):
    anchors = np.arange(14, len(uni), dtype=np.int64)
    ends = uni[anchors]
    bad_steps = np.r_[0, np.cumsum(np.diff(uni) != 60)]
    continuous = (bad_steps[anchors] - bad_steps[anchors - 14]) == 0
    targets = np.searchsorted(uni, ends + horizon * 60)
    safe = np.minimum(targets, len(uni) - 1)
    valid = continuous & (targets < len(uni)) & (uni[safe] == ends + horizon * 60)
    anchors, targets = anchors[valid], targets[valid]
    return dict(anchors=anchors, win=anchors[:, None] + np.arange(-14, 1),
                tgt=targets, anchor_time=uni[anchors].astype('datetime64[s]'))


def prepare_split(split, source, output, write_images=True):
    paths = raw_files(source, split)
    uni, frame_rows, pv, source_rows, n_raw = metadata(paths)
    output.mkdir(parents=True, exist_ok=True)
    if write_images:
        mm = np.lib.format.open_memmap(output / f'{split}_images.dat.npy', mode='w+',
                                      dtype=np.uint8, shape=(len(uni), 64, 64, 3))
        dest = np.full(n_raw, -1, dtype=np.int64)
        dest[source_rows] = np.arange(len(uni))
        offset = 0
        for path in paths:
            for batch in batches(path):
                image_column = batch.column(batch.schema.get_field_index('image'))
                for j, image_value in enumerate(image_column):
                    row = dest[offset + j]
                    if row < 0:
                        continue
                    blob = image_value.as_py()['bytes']
                    if blob is None:
                        raise ValueError('Expected embedded upstream image bytes')
                    with Image.open(io.BytesIO(blob)) as image:
                        arr = np.asarray(image.convert('RGB'), dtype=np.uint8)
                    if arr.shape != (64, 64, 3):
                        raise ValueError(f'Unexpected frame shape {arr.shape}')
                    mm[row] = arr
                offset += len(batch)
            print(f'{split}: decoded {offset}/{n_raw}', flush=True)
        mm.flush()
        del mm
    np.savez(output / f'{split}_meta.npz', minutes=uni.astype('datetime64[s]'),
             frame_rows=frame_rows, pv=pv)
    for horizon in (5, 15, 30):
        suffix = '' if horizon == 15 else f'_h{horizon}'
        samples = build_windows(uni, horizon)
        np.savez(output / f'{split}_samples{suffix}.npz', **samples)
        print(f'{split}: horizon={horizon} windows={len(samples["tgt"])}', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('splits', nargs='*', default=['test', 'train'], metavar='SPLIT')
    default_source = ROOT / 'data/My_SKIPPD_Data'
    if not default_source.exists():
        default_source = ROOT / 'data/raw'
    ap.add_argument('--source', type=Path, default=default_source)
    ap.add_argument('--output', type=Path, default=ROOT / 'data/proc')
    ap.add_argument('--metadata-only', action='store_true')
    args = ap.parse_args()
    if any(split not in ['train', 'test'] for split in args.splits):
        ap.error('SPLIT must be train or test')
    for split in args.splits:
        prepare_split(split, args.source, args.output, not args.metadata_only)


if __name__ == '__main__':
    main()
