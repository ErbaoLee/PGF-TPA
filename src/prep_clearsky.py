# -*- coding: utf-8 -*-
"""预计算每分钟晴空理论功率 P_theo（官方 Relative_op 解析模型，kW）。
输出 data/proc/{split}_clearsky.npz: P_theo(M,) 与分钟索引对齐（meta.minutes）。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
import importlib.util

spec = importlib.util.spec_from_file_location(
    "rel_op", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "relative_op_func_reference.py"))
rel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rel)

for split in ["train", "test"]:
    meta = np.load(os.path.join(C.PROC, f"{split}_meta.npz"))
    minutes = meta["minutes"]
    out = np.zeros(len(minutes), dtype=np.float64)
    for i, m in enumerate(minutes):
        t = m.astype("datetime64[s]").astype(object)
        try:
            _, p = rel.Relative_output(t, 1.0)
            out[i] = p
        except Exception:
            out[i] = np.nan
    nan = np.isnan(out).sum()
    if nan:
        out = np.where(np.isnan(out), np.nanmedian(out), out)
    np.savez(os.path.join(C.PROC, f"{split}_clearsky.npz"), p_theo=out)
    print(f"{split}: P_theo done, n={len(out)}, nan_fixed={nan}, "
          f"range=[{out.min():.2f},{out.max():.2f}]")
