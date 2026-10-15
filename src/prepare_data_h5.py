# -*- coding: utf-8 -*-
"""生成 5 分钟尺度的 forecast 窗口索引（协议 v1.2 扩展，见
BENCHMARK_LOCK_v1.2_extension.md）。

规则与 prepare_data.py 的 15 分钟版完全同构，仅目标位移 s+900 → s+300：
  锚点 t（分钟），输入=[t-14,t] 15 帧严格连续 60s，目标=t+5min 的 pv。
输入直接复用 {split}_meta.npz（分钟序列 + pv + 帧行号），不重新解码图像。
输出：data/proc/{split}_samples_h5.npz (anchors, win, tgt, anchor_time)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

PROC = C.PROC


def build_h5(split, horizon_min=5):
    meta = np.load(os.path.join(PROC, f"{split}_meta.npz"))
    uni = meta["minutes"].astype(np.int64)  # 秒
    pos = {s: k for k, s in enumerate(uni)}
    off = horizon_min * 60

    anchors, wins, tgts = [], [], []
    for k, s in enumerate(uni):
        j = k - 14
        if j < 0:
            continue
        tgt = pos.get(s + off)
        if tgt is None:
            continue
        if (uni[j:k + 1] == np.arange(s - 840, s + 60, 60)).all():
            anchors.append(k)
            wins.append(np.arange(j, k + 1))
            tgts.append(tgt)
    anchors = np.array(anchors, dtype=np.int64)
    wins = np.array(wins, dtype=np.int64)
    tgts = np.array(tgts, dtype=np.int64)
    anchor_time = uni[anchors].astype("datetime64[s]")

    out = os.path.join(PROC, f"{split}_samples_h{horizon_min}.npz")
    np.savez(out, anchors=anchors, win=wins, tgt=tgts, anchor_time=anchor_time)
    print(f"{split}: h{horizon_min} samples={len(anchors)} -> {out}")


if __name__ == "__main__":
    # 用法：python prepare_data_h5.py [horizon_min] [split...]
    #   例：python prepare_data_h5.py            # 默认 5 min，test+train
    #       python prepare_data_h5.py 30         # 30 min，test+train
    #       python prepare_data_h5.py 5 test     # 只生成 test
    argv = sys.argv[1:]
    horizon = 5
    if argv and argv[0].isdigit():
        horizon = int(argv[0])
        argv = argv[1:]
    for split in (argv or ["test", "train"]):
        build_h5(split, horizon_min=horizon)
    print("DONE")
