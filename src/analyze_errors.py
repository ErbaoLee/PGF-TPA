# -*- coding: utf-8 -*-
"""误差分解分析：验证 M1(CSI 参数化)/M2(云运动)/M3(晴阴门控) 的触发条件。
只用 seed0 预测 + 锁定 test——不做任何模型选择，仅诊断。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

te = C.get_split("test")
at = te["anchor_time"]
y = te["pv"][te["tgt"]].astype(np.float64)
pv_log = te["pv"][te["win"]].astype(np.float64)
sunny, cloudy = C.sunny_cloudy_masks(at)
hours = np.array([np.datetime64(t, "s").astype(object).hour for t in at])

preds = {}
for m in ["mlp", "gru", "sunset"]:
    p = os.path.join(C.RESULTS_DIR, f"preds_{m}_s0.npy")
    if os.path.exists(p):
        preds[m] = np.load(p)
preds["raw_per"] = pv_log[:, -1]

def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))

print("== 1) 晴/阴 × 时段 残差 RMSE ==")
bins = [(6, 9), (9, 12), (12, 15), (15, 19)]
for name, p in preds.items():
    row = []
    for lo, hi in bins:
        m = sunny & (hours >= lo) & (hours < hi)
        row.append(f"{rmse(p[m], y[m]):.2f}" if m.sum() > 30 else "-")
    print(f"  {name:8s} sunny RMSE by hour {bins}: {row}")

print("\n== 2) 阴天 ramp 条件误差 ==")
ramp = np.abs(y - pv_log[:, -1])  # 15 分钟目标变化幅度
qs = np.quantile(ramp[cloudy], [0.5, 0.8, 0.95])
print(f"  cloudy |ΔPV| quantiles: p50={qs[0]:.2f} p80={qs[1]:.2f} p95={qs[2]:.2f} kW")
for name, p in preds.items():
    m1 = cloudy & (ramp <= qs[0])
    m2 = cloudy & (ramp >= qs[1])
    print(f"  {name:8s} cloudy RMSE: stable(Δ≤p50)={rmse(p[m1], y[m1]):.2f} "
          f"ramp(Δ≥p80)={rmse(p[m2], y[m2]):.2f} (n={m1.sum()}/{m2.sum()})")

print("\n== 3) 晴天系统性偏差（M1 触发条件）==")
for name, p in preds.items():
    m = sunny
    bias = float(np.mean(p[m] - y[m]))
    mae = float(np.mean(np.abs(p[m] - y[m])))
    print(f"  {name:8s} sunny bias={bias:+.3f} kW, MAE={mae:.3f} -> "
          f"bias占比 {abs(bias)/mae*100:.0f}%")

print("\n== 4) 融合上界（oracle min-error 选择，M3 触发条件）==")
if "gru" in preds and "sunset" in preds:
    oracle = np.minimum(np.abs(preds["gru"] - y), np.abs(preds["sunset"] - y))
    # oracle RMSE：逐样本选误差小的模型后合成（用其预测值）
    pick = np.abs(preds["gru"] - y) <= np.abs(preds["sunset"] - y)
    yhat_or = np.where(pick, preds["gru"], preds["sunset"])
    print(f"  GRU overall RMSE   = {rmse(preds['gru'], y):.3f}")
    print(f"  SUNSET overall RMSE= {rmse(preds['sunset'], y):.3f}")
    print(f"  ORACLE(per-sample) = {rmse(yhat_or, y):.3f}")
    print(f"  pick-GRU share: sunny {pick[sunny].mean():.2f}, cloudy {pick[cloudy].mean():.2f}")
    # 简单门控：按窗口内 PV 方差选择
    var_w = pv_log.var(axis=1)
    thr = np.quantile(var_w, 0.5)
    gate_gru = var_w <= thr  # 稳定窗口用 GRU
    yhat_g = np.where(gate_gru, preds["gru"], preds["sunset"])
    print(f"  VAR-GATE(用pv_log方差阈值) RMSE = {rmse(yhat_g, y):.3f}")
    print(f"  简单均值融合 RMSE = {rmse((preds['gru']+preds['sunset'])/2, y):.3f}")
