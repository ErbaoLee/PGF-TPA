# -*- coding: utf-8 -*-
"""协议对齐检验：由镜像序列重建 forecast 样本，复算 persistence 基准，
与官方论文（Nie 2023, FS 反推）对比。这是是否可以开训的门槛。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

DATA = os.path.join(C.DATA_DIR, "My_SKIPPD_Data")

from datasets import load_from_disk  # noqa: E402

ds = load_from_disk(DATA)
frames = {}
for split in ["train", "test"]:
    d = ds[split]
    t = pd.Series(pd.to_datetime([str(x) for x in d["time"]], utc=True)
                  ).dt.tz_convert("US/Pacific").sort_values()
    order = np.argsort(t.values)
    pv = np.asarray(d["pv"], dtype=np.float64)[order]
    t = t.iloc[order].reset_index(drop=True)
    frames[split] = (t, pv)
    gaps = np.diff(t.values).astype("timedelta64[s]").astype(int)
    print(f"{split}: n={len(t)} {t.iloc[0]} -> {t.iloc[-1]} "
          f"frac60s={(gaps==60).mean():.4f} maxgap={gaps.max()}s")

def build_samples(t, pv, log_len=15, horizon=15):
    """窗口对齐官方 forecast：对每个锚点 i（序列位置），
    pv_log = pv[i-log_len+1 .. i]（连续 1-min），pv_pred = 锚点后 horizon 分钟的 pv。
    时间戳先 round 到整分（镜像含相机帧秒偏移），窗口按分钟连续性判定。"""
    tmin = t.dt.round("min")
    sec = tmin.values.astype("datetime64[s]").astype(np.int64)
    lookup = {}
    for k, s in enumerate(sec):
        lookup.setdefault(s, k)  # 同一分钟多帧时取第一帧
    uni = np.array(sorted(lookup.keys()))
    pv_min = np.array([pv[lookup[s]] for s in uni], dtype=np.float64)
    anchors = []
    pv_logs, pv_preds, times_anchor = [], [], []
    pos = {s: k for k, s in enumerate(uni)}
    for k, s in enumerate(uni):
        j = k - log_len + 1
        if j < 0:
            continue
        if (uni[j:k + 1] == np.arange(s - (log_len - 1) * 60, s + 60, 60)).all() \
                and pos.get(s + horizon * 60) is not None:
            pv_logs.append(pv_min[j:k + 1])
            pv_preds.append(pv_min[pos[s + horizon * 60]])
            times_anchor.append(tmin.iloc[k])
    return (np.array(times_anchor, dtype=object), np.array(pv_logs), np.array(pv_preds))

for split in ["test"]:
    t, pv = frames[split]
    ta, pv_log, pv_pred = build_samples(t, pv)
    print(f"\n{split} rebuilt forecast samples: n={len(ta)} (官方 14003)")

    sunny, cloudy = C.sunny_cloudy_masks(ta)
    print("sunny n:", sunny.sum(), "cloudy n:", cloudy.sum())

    # raw persistence
    rp = pv_log[:, -1]
    res_rp = C.evaluate(pv_pred, rp, ta)
    print("RAW persistence RMSE: sunny %.3f cloudy %.3f overall %.3f"
          % (res_rp["sunny"]["rmse"], res_rp["cloudy"]["rmse"], res_rp["overall"]["rmse"]))

    # CSI persistence（官方实现）
    csi = C.csi_persistence(ta, pv_log, pv_pred)
    res_csi = C.evaluate(pv_pred, csi, ta)
    print("CSI persistence RMSE: sunny %.3f cloudy %.3f overall %.3f"
          % (res_csi["sunny"]["rmse"], res_csi["cloudy"]["rmse"], res_csi["overall"]["rmse"]))
    print("\n官方反推 CSI-persistence RMSE 目标: sunny≈0.42 cloudy≈5.15 overall≈3.63 (kW)")
